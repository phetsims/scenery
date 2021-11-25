// Copyright 2021, University of Colorado Boulder

/**
 * Interface specifically for SelfDrawables for a Path Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

interface IPathDrawable {
  markPaintDirty(): void;
  markDirtyShape(): void;
}

export default IPathDrawable;