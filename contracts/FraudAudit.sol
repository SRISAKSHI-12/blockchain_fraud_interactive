// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract FraudAudit {
    struct AuditRecord {
        bytes32 transactionHash;
        bool fraudPrediction;
        uint16 riskScore;
        string riskLevel;
        bytes32 explanationHash;
        string action;
        string modelVersion;
        uint256 timestamp;
    }
    AuditRecord[] public records;
    event RecordAdded(uint256 indexed id, bytes32 indexed transactionHash, bool fraudPrediction, uint16 riskScore, string riskLevel);

    function addRecord(
        bytes32 transactionHash, bool fraudPrediction, uint16 riskScore,
        string calldata riskLevel, bytes32 explanationHash,
        string calldata action, string calldata modelVersion
    ) external {
        require(riskScore <= 100, "Invalid risk score");
        records.push(AuditRecord(transactionHash,fraudPrediction,riskScore,riskLevel,explanationHash,action,modelVersion,block.timestamp));
        emit RecordAdded(records.length-1,transactionHash,fraudPrediction,riskScore,riskLevel);
    }
    function recordCount() external view returns(uint256){ return records.length; }
}
