use super::{Contract, ContractError, Observation, Term};
use crate::{Fingerprint, JsonLimits};
use serde::{Deserialize, Serialize};

pub(super) mod date {
    use crate::core::Date;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(value: &Date, s: S) -> Result<S::Ok, S::Error> {
        value.to_string().serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Date, D::Error> {
        String::deserialize(d)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}
pub(super) mod dates {
    use crate::core::Date;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(value: &[Date], serializer: S) -> Result<S::Ok, S::Error> {
        value
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .serialize(serializer)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<Date>, D::Error> {
        Vec::<String>::deserialize(d)?
            .into_iter()
            .map(|s| s.parse().map_err(serde::de::Error::custom))
            .collect()
    }
}
macro_rules! id_codec {
    ($module:ident,$id:ident,$raw:ty) => {
        pub(super) mod $module {
            use crate::core::$id;
            use serde::{Deserialize, Deserializer, Serialize, Serializer};
            pub fn serialize<S: Serializer>(value: &$id, s: S) -> Result<S::Ok, S::Error> {
                value.get().serialize(s)
            }
            pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<$id, D::Error> {
                <$raw>::deserialize(d).map($id::new)
            }
        }
    };
}
id_codec!(underlying, UnderlyingId, u32);
id_codec!(currency, CurrencyId, u16);
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    document_kind: String,
    schema_version: u32,
    id: String,
    observations: Vec<Observation>,
    root: Term,
}
impl Contract {
    pub(super) fn schema_version(&self) -> u32 {
        let mut options = Vec::new();
        self.root.visit_options(&mut options);
        if options.is_empty() { 1 } else { 2 }
    }
    pub fn to_json(&self) -> Result<String, ContractError> {
        self.validate()?;
        serde_json::to_string(&Document {
            document_kind: "contract".into(),
            schema_version: self.schema_version(),
            id: self.id.clone(),
            observations: self.observations.clone(),
            root: self.root.clone(),
        })
        .map_err(|e| ContractError::Wire(e.to_string()))
    }
    pub fn from_json(input: &[u8], limits: JsonLimits) -> Result<Self, ContractError> {
        let text = crate::wire::validate_and_decode(input, limits)
            .map_err(|e| ContractError::Wire(e.to_string()))?;
        let d: Document =
            serde_json::from_str(text).map_err(|e| ContractError::Wire(e.to_string()))?;
        if d.document_kind != "contract" || !(1..=2).contains(&d.schema_version) {
            return Err(ContractError::Wire(
                "expected contract schema version 1 or 2".into(),
            ));
        }
        let contract = Self::new(d.id, d.observations, d.root)?;
        if contract.schema_version() != d.schema_version {
            return Err(ContractError::Wire(
                "contract features do not match the declared version".into(),
            ));
        }
        Ok(contract)
    }
    pub fn fingerprint(&self) -> Result<Fingerprint, ContractError> {
        let mut bytes = format!("pricing/contract/v{}\0", self.schema_version()).into_bytes();
        bytes.extend_from_slice(self.to_json()?.as_bytes());
        Ok(Fingerprint::from_bytes(*blake3::hash(&bytes).as_bytes()))
    }
}

impl Contract {
    pub fn json_schema() -> &'static str {
        include_str!("../../../../schemas/contract-v1/contract.schema.json")
    }
    pub fn json_schema_for_version(version: u32) -> Result<&'static str, ContractError> {
        match version {
            1 => Ok(Self::json_schema()),
            2 => Ok(include_str!(
                "../../../../schemas/contract-v2/contract.schema.json"
            )),
            _ => Err(ContractError::Wire(
                "unknown Contract schema version".into(),
            )),
        }
    }
    pub fn analysis_json(&self) -> Result<String, ContractError> {
        let a = self.analyze()?;
        let mut v = serde_json::json!({"document_kind":"contract_analysis","schema_version":1,
            "required_observations":a.required_observations,"potential_cashflows":a.potential_cashflows,
            "determined_cashflows":a.determined_cashflows,"underlying_ids":a.underlying_dependencies.iter().map(|u|u.get()).collect::<Vec<_>>(),
            "currency_ids":a.currencies.iter().map(|c|c.get()).collect::<Vec<_>>(),"barrier_monitoring_dates":a.barrier_monitoring_dates.iter().map(ToString::to_string).collect::<Vec<_>>()});
        if !a.option_features.is_empty() {
            v["option_features"] = serde_json::json!(a.option_features);
            v["schema_version"] = serde_json::json!(2);
        }
        if let Some(date) = a.horizon {
            v["horizon"] = serde_json::json!(date.to_string());
        }
        serde_json::to_string(&v).map_err(|e| ContractError::Wire(e.to_string()))
    }
}

impl Contract {
    pub fn observations_json(&self) -> Result<String, ContractError> {
        serde_json::to_string(self.observations()).map_err(|e| ContractError::Wire(e.to_string()))
    }
}
