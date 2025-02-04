// Copyright 2022-2025, University of Colorado Boulder

/**
 * Interface specifically for SelfDrawables for a Line Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
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