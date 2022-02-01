// Copyright 2021, University of Colorado Boulder

/**
 * An AlignBox that syncs its alignBounds to a specific Bounds2 Property.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import IReadOnlyProperty from '../../../axon/js/IReadOnlyProperty.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import merge from '../../../phet-core/js/merge.js';
import { scenery, AlignBox, AlignBoxOptions, Node } from '../imports.js';

type AlignPropertyBoxOptions = AlignBoxOptions;

class AlignPropertyBox extends AlignBox {

  private alignBoundsProperty: IReadOnlyProperty<Bounds2>;
  private _alignBoundsPropertyListener: ( b: Bounds2 ) => void;

  /**
   * @param content - Content to align inside the alignBox
   * @param alignBoundsProperty
   * @param [options]
   */
  constructor( content: Node, alignBoundsProperty: IReadOnlyProperty<Bounds2>, providedOptions?: AlignPropertyBoxOptions ) {
    const options = merge( {
      alignBounds: alignBoundsProperty.value
    }, providedOptions );

    super( content, options );

    this._alignBoundsPropertyListener = ( bounds: Bounds2 ) => { this.alignBounds = bounds; };
    this.alignBoundsProperty = alignBoundsProperty;
    this.alignBoundsProperty.lazyLink( this._alignBoundsPropertyListener );
  }

  /**
   * Releases references
   */
  dispose() {
    this.alignBoundsProperty.unlink( this._alignBoundsPropertyListener );

    super.dispose();
  }
}

scenery.register( 'AlignPropertyBox', AlignPropertyBox );
export default AlignPropertyBox;
export type { AlignPropertyBoxOptions };
