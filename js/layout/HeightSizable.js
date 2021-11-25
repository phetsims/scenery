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

const HEIGHT_SIZABLE_OPTION_KEYS = [
  'preferredHeight',
  'minimumHeight'
];

const HeightSizable = memoize( type => {
  const clazz = class extends type {
    constructor( ...args ) {
      super( ...args );

      // @public {Property.<number|null>}
      this.preferredHeightProperty = new TinyProperty( null );
      this.minimumHeightProperty = new TinyProperty( null );

      // @public {boolean} - Flag for detection of the feature
      this.heightSizable = true;
    }

    /**
     * @public
     *
     * @returns {number|null}
     */
    get preferredHeight() {
      return this.preferredHeightProperty.value;
    }

    /**
     * @public
     *
     * @param {number|null} value
     */
    set preferredHeight( value ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ),
        'preferredHeight should be null or a non-negative finite number' );

      this.preferredHeightProperty.value = value;
    }

    /**
     * @public
     *
     * @returns {number|null}
     */
    get minimumHeight() {
      return this.minimumHeightProperty.value;
    }

    /**
     * @public
     *
     * @param {number|null} value
     */
    set minimumHeight( value ) {
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