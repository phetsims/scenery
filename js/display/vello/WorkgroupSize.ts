// Copyright 2023, University of Colorado Boulder

import { scenery } from '../../imports.js';

/**
 * Stores a workgroup size (x, y, z) for a compute shader.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

export class WorkgroupSize {
  // u32 in rust
  public constructor( public x: number, public y: number, public z: number ) {
  }

  public toString(): string {
    return `[${this.x} ${this.y} ${this.z}]`;
  }
}

scenery.register( 'WorkgroupSize', WorkgroupSize );