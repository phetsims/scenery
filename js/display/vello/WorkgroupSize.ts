// Copyright 2023, University of Colorado Boulder

import { scenery } from '../../imports.js';

/**
 * Stores a workgroup size (x, y, z) for a compute shader.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

export default class WorkgroupSize {
  // u32 in rust
  public constructor(
    public readonly x: number,
    public readonly y: number,
    public readonly z: number
  ) {}

  public toString(): string {
    return `[${this.x} ${this.y} ${this.z}]`;
  }
}

scenery.register( 'WorkgroupSize', WorkgroupSize );
