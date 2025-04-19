CREATE DATABASE ContinualRelationExtraction;
USE ContinualRelationExtraction;

CREATE TABLE relation_records
(
    id          INT AUTO_INCREMENT PRIMARY KEY,
    sentence    TEXT         NOT NULL,
    entity1     VARCHAR(255) NOT NULL,
    entity2     VARCHAR(255) NOT NULL,
    relation    VARCHAR(100) NOT NULL,
    confidence  FLOAT        NOT NULL,
    create_time DATETIME     NOT NULL,
    INDEX idx_entities (entity1, entity2),
    INDEX idx_relation (relation),
    INDEX idx_time (create_time)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4;
