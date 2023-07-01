// Copyright 2023, University of Colorado Boulder

/**
 * An image from a buffer with a width/height
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../../imports.js';
import IntentionalAny from '../../../../phet-core/js/types/IntentionalAny.js';

export default class BufferImage {
  // TODO: perhaps reorder parameters
  // NOTE: IMPORTANT! If using this, make sure the buffer has premultiplied values. Canvas.getImageData() is NOT
  // premultiplied. Use SourceImage with Canvases
  public constructor(
    public readonly width: number,
    public readonly height: number,
    public readonly buffer: ArrayBuffer
  ) {
    assert && assert( isFinite( width ) && width >= 0 );
    assert && assert( isFinite( height ) && height >= 0 );
  }

  public equals( other: IntentionalAny ): boolean {
    return other instanceof BufferImage && this.buffer === other.buffer;
  }
}

scenery.register( 'BufferImage', BufferImage );
