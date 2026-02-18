// Copyright 2022-2026, University of Colorado Boulder

/**
 * Interface specifically for SelfDrawables for a Line Node.
 *
 * @author Jonathan Olson (PhET Interactive Simulations)
 */

import type TPathDrawable from '../../display/drawables/TPathDrawable.js';

type TLineDrawable = {
  markDirtyLine(): void;
  markDirtyP1(): void;
  markDirtyP2(): void;
  markDirtyX1(): void;
  markDirtyY1(): void;
  markDirtyX2(): void;
  markDirtyY2(): void;
} & TPathDrawable;

export default TLineDrawable;