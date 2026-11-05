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
    pub fn to_json(&self) -> Result<String, ContractError> {
        self.validate()?;
        serde_json::to_string(&Document {
            document_kind: "contract".into(),
            schema_version: 1,
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
        if d.document_kind != "contract" || d.schema_version != 1 {
            return Err(ContractError::Wire(
                "expected contract schema version 1".into(),
            ));
        }
        Self::new(d.id, d.observations, d.root)
    }
    pub fn fingerprint(&self) -> Result<Fingerprint, ContractError> {
        let mut bytes = b"pricing/contract/v1\0".to_vec();
        bytes.extend_from_slice(self.to_json()?.as_bytes());
        Ok(Fingerprint::from_bytes(*blake3::hash(&bytes).as_bytes()))
    }
}
