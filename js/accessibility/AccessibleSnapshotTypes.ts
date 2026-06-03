// Copyright 2026, University of Colorado Boulder

/**
 * Snapshot contracts used by agent-facing accessibility tooling. These are intentionally sparse, semantic object
 * shapes rather than serialized model/view graphs.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */

export type AccessibleSnapshotPrimitive = string | number | boolean | null;

export type AccessibleSnapshotValue =
  AccessibleSnapshotPrimitive |
  AccessibleSnapshotValue[] |
  { [ key: string ]: AccessibleSnapshotValue | undefined };

export type AccessibleStateNode = {
  type?: string;
  name?: string;
  children?: AccessibleStateNode[];
  [ key: string ]: AccessibleSnapshotValue | AccessibleStateNode[] | undefined;
};

export type AccessibleViewStateNode = {
  [ key: string ]: AccessibleSnapshotValue | AccessibleViewStateNode | AccessibleViewStateNode[] | undefined;
};
