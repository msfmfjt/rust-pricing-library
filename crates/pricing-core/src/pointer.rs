use std::fmt;
use std::str::FromStr;

use crate::CoreError;

/// A validated RFC 6901 JSON Pointer. The empty string identifies the root.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct JsonPointer(String);

impl JsonPointer {
    #[must_use]
    pub fn root() -> Self {
        Self(String::new())
    }

    pub fn new(value: impl Into<String>) -> Result<Self, CoreError> {
        let value = value.into();
        validate(&value)?;
        Ok(Self(value))
    }

    #[must_use]
    pub fn from_tokens<'a, I>(tokens: I) -> Self
    where
        I: IntoIterator<Item = &'a str>,
    {
        let mut pointer = String::new();
        for token in tokens {
            pointer.push('/');
            for character in token.chars() {
                match character {
                    '~' => pointer.push_str("~0"),
                    '/' => pointer.push_str("~1"),
                    _ => pointer.push(character),
                }
            }
        }
        Self(pointer)
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for JsonPointer {
    fn default() -> Self {
        Self::root()
    }
}

impl fmt::Display for JsonPointer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl FromStr for JsonPointer {
    type Err = CoreError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

fn validate(value: &str) -> Result<(), CoreError> {
    if !value.is_empty() && !value.starts_with('/') {
        return Err(CoreError::InvalidJsonPointer {
            value: value.to_owned(),
        });
    }
    let bytes = value.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'~' {
            if index + 1 >= bytes.len() || !matches!(bytes[index + 1], b'0' | b'1') {
                return Err(CoreError::InvalidJsonPointer {
                    value: value.to_owned(),
                });
            }
            index += 2;
        } else {
            index += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_and_tokens_follow_rfc_6901() {
        assert_eq!(JsonPointer::root().as_str(), "");
        let pointer = JsonPointer::from_tokens(["products", "a/b", "x~y"]);
        assert_eq!(pointer.as_str(), "/products/a~1b/x~0y");
        assert_eq!(pointer.as_str().parse(), Ok(pointer));
    }

    #[test]
    fn malformed_pointers_are_rejected() {
        assert!("products/0".parse::<JsonPointer>().is_err());
        assert!("/products/~".parse::<JsonPointer>().is_err());
        assert!("/products/~2".parse::<JsonPointer>().is_err());
    }
}
