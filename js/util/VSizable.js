// Copyright 2021, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyProperty from '../../../axon/js/TinyProperty.js';
import memoize from '../../../phet-core/js/memoize.js';
import scenery from '../scenery.js';

const V_SIZABLE_OPTION_KEYS = [
  'preferredWidth',
  'preferredHeight',
  'minimumWidth',
  'minimumHeight'
];

const VSizable = memoize( type => {
  const clazz = class extends type {
    constructor( ...args ) {
      super( ...args );

      // @public {Property.<number|null>}
      this.preferredHeightProperty = new TinyProperty( null );
      this.minimumHeightProperty = new TinyProperty( null );

      // @public {boolean} - Flag for detection of the feature
      this.vSizable = true;
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
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

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
    clazz.prototype._mutatorKeys = type.prototype._mutatorKeys.concat( V_SIZABLE_OPTION_KEYS );
  }

  return clazz;
} );

scenery.register( 'VSizable', VSizable );
export default VSizable;