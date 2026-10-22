//! Runtime validation of the mrec record schema.
//!
//! The language-neutral JSON Schema is published by molrec
//! (`schema/core/record.schema.json`). This module is the executable form
//! molrs runs on a record's snapshot and system-definition frames.

use molrs::core::Frame;
use molrs::core::MolRsError;

/// Judge a snapshot or system-definition frame against the Frame vocabulary.
pub fn validate_frame(frame: &Frame) -> Result<(), MolRsError> {
    frame.validate()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_frame_passes_vocabulary() {
        validate_frame(&Frame::new()).unwrap();
    }
}
