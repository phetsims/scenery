// Copyright 2021, University of Colorado Boulder

/**
 * Provides a minimum and preferred height. The minimum height is set by the component, so that layout containers could
 * know how "small" the component can be made. The preferred height is set by the layout container, and the component
 * should adjust its size so that it takes up that height.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyProperty from '../../../axon/js/TinyProperty.js';
import memoize from '../../../phet-core/js/memoize.js';
import { scenery } from '../imports.js';
import Constructor from '../../../phet-core/js/Constructor.js';

const HEIGHT_SIZABLE_OPTION_KEYS = [
  'preferredHeight',
  'minimumHeight'
];

const HeightSizable = memoize( <SuperType extends Constructor>( type: SuperType ) => {
  const clazz = class extends type {

    preferredHeightProperty: TinyProperty<number | null>;
    minimumHeightProperty: TinyProperty<number | null>;

    // Flag for detection of the feature
    heightSizable: boolean;

    constructor( ...args: any[] ) {
      super( ...args );

      this.preferredHeightProperty = new TinyProperty<number | null>( null );
      this.minimumHeightProperty = new TinyProperty<number | null >( null );
      this.heightSizable = true;
    }

    get preferredHeight(): number | null {
      return this.preferredHeightProperty.value;
    }

    set preferredHeight( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ),
        'preferredHeight should be null or a non-negative finite number' );

      this.preferredHeightProperty.value = value;
    }

    get minimumHeight(): number | null {
      return this.minimumHeightProperty.value;
    }

    set minimumHeight( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      this.minimumHeightProperty.value = value;
    }
  };

  // If we're extending into a Node type, include option keys
  // TODO: This is ugly, we'll need to mutate after construction, no?
  if ( type.prototype._mutatorKeys ) {
    clazz.prototype._mutatorKeys = type.prototype._mutatorKeys.concat( HEIGHT_SIZABLE_OPTION_KEYS );
  }

  return clazz;
} );

scenery.register( 'HeightSizable', HeightSizable );
export default HeightSizable;