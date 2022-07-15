// Copyright 2022, University of Colorado Boulder

/**
 * An AlignBox that syncs its alignBounds to a specific Bounds2 Property.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import IReadOnlyProperty from '../../../../axon/js/IReadOnlyProperty.js';
import Bounds2 from '../../../../dot/js/Bounds2.js';
import optionize from '../../../../phet-core/js/optionize.js';
import EmptyObjectType from '../../../../phet-core/js/types/EmptyObjectType.js';
import { AlignBox, AlignBoxOptions, Node, scenery } from '../../imports.js';

export type AlignPropertyBoxOptions = AlignBoxOptions;

export default class AlignPropertyBox extends AlignBox {

  private readonly alignBoundsProperty: IReadOnlyProperty<Bounds2>;
  private readonly _alignBoundsPropertyListener: ( b: Bounds2 ) => void;

  /**
   * @param content - Content to align inside the alignBox
   * @param alignBoundsProperty
   * @param [providedOptions]
   */
  public constructor( content: Node, alignBoundsProperty: IReadOnlyProperty<Bounds2>, providedOptions?: AlignPropertyBoxOptions ) {
    const options = optionize<AlignPropertyBoxOptions, EmptyObjectType, AlignBoxOptions>()( {
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
  public override dispose(): void {
    this.alignBoundsProperty.unlink( this._alignBoundsPropertyListener );

    super.dispose();
  }
}

scenery.register( 'AlignPropertyBox', AlignPropertyBox );
