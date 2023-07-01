// Copyright 2023, University of Colorado Boulder

/**
 * An image from a Canvas/ImageBitmap with a width/height.
 *
 * This is particularly useful to avoid all sorts of computations/overhead for getting an immediately-writable
 * premultiplied image. Canvas.getImageData() is NOT premultiplied, so we can't easily use BufferImage to copy things
 * over. With this type, we can use things that get copied over very efficiently, without more Canvas draws or other
 * hackery.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../../imports.js';
import IntentionalAny from '../../../../phet-core/js/types/IntentionalAny.js';

export default class SourceImage {
  // TODO: perhaps reorder parameters
  public constructor(
    public readonly width: number,
    public readonly height: number,
    public readonly source: GPUImageCopyExternalImageSource
  ) {
    assert && assert( isFinite( width ) && width >= 0 );
    assert && assert( isFinite( height ) && height >= 0 );
  }

  public equals( other: IntentionalAny ): boolean {
    return other instanceof SourceImage && this.source === other.source;
  }
}

scenery.register( 'SourceImage', SourceImage );
