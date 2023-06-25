// Copyright 2023, University of Colorado Boulder

/**
 * An image from a buffer with a width/height
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../../imports.js';

export class BufferImage {
  // TODO: perhaps reorder parameters
  // NOTE: IMPORTANT! If using this, make sure the buffer has premultiplied values. Canvas.getImageData() is NOT
  // premultiplied. Use SourceImage with Canvases
  public constructor(
    public readonly width: number,
    public readonly height: number,
    public readonly buffer: ArrayBuffer
  ) {}
}

scenery.register( 'BufferImage', BufferImage );
