// Copyright 2021, University of Colorado Boulder

/**
 * An AlignBox that syncs its alignBounds to a specific Bounds2 Property.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import merge from '../../../phet-core/js/merge.js';
import { scenery, AlignBox } from '../imports.js';

class AlignPropertyBox extends AlignBox {
  /**
   * @param {Node} content - Content to align inside of the alignBox
   * @param {Property.<Bounds2>} alignBoundsProperty
   * @param {Object} [options]
   */
  constructor( content, alignBoundsProperty, options ) {
    options = merge( {
      alignBounds: alignBoundsProperty.value
    }, options );

    super( content, options );

    // @private {function}
    this._alignBoundsPropertyListener = bounds => { this.alignBounds = bounds; };

    // @private {Property.<Bounds2>}
    this.alignBoundsProperty = alignBoundsProperty;

    this.alignBoundsProperty.lazyLink( this._alignBoundsPropertyListener );
  }

  /**
   * Releases references
   * @public
   * @override
   */
  dispose() {
    this.alignBoundsProperty.unlink( this._alignBoundsPropertyListener );

    super.dispose();
  }
}

scenery.register( 'AlignPropertyBox', AlignPropertyBox );
export default AlignPropertyBox;