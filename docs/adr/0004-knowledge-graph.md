# ADR 0004: Knowledge Graph

## Problem
In forensic video investigations, objects (cars, people, license plates) do not exist in isolation. They interact across time and space. A standard vector database retrieves semantically similar frames but cannot answer graph traversal queries (e.g., "Which camera saw the white van after it left the parking lot?").

## Decision
We implemented a property-based `KnowledgeGraph` domain model. All entities extracted by the Video and Metadata agents are immediately projected into nodes, and their temporal/spatial interactions are stored as edges.

## Alternatives Considered
- **Relational DB (Postgres) only**: (Too slow and complex for multi-hop spatial-temporal queries).
- **Vector DB only**: (Fails at topological traversal and exact logical joins).

## Consequences
- **Positive**: Enables the Reasoning Engine to perform multi-hop logic and identify missing spatial gaps, allowing for highly complex investigations.
- **Negative**: Requires maintaining two data layers (Vector + Graph) and projecting unstructured data into strict schema properties.
