// Copyright 2023, University of Colorado Boulder

/**
 * An image from a buffer with a width/height
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../../imports.js';

export class BufferImage {
  // TODO: perhaps reorder parameters
  public constructor(
    public readonly width: number,
    public readonly height: number,
    public readonly buffer: ArrayBuffer
  ) {}
}

scenery.register( 'BufferImage', BufferImage );
