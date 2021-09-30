// Copyright 2021, University of Colorado Boulder

/**
 * Provides a minimum and preferred width. The minimum width is set by the component, so that layout containers could
 * know how "small" the component can be made. The preferred width is set by the layout container, and the component
 * should adjust its size so that it takes up that width.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyProperty from '../../../axon/js/TinyProperty.js';
import memoize from '../../../phet-core/js/memoize.js';
import scenery from '../scenery.js';

const WIDTH_SIZABLE_OPTION_KEYS = [
  'preferredWidth',
  'minimumWidth'
];

const WidthSizable = memoize( type => {
  const clazz = class extends type {
    constructor( ...args ) {
      super( ...args );

      // @public {Property.<number|null>}
      this.preferredWidthProperty = new TinyProperty( null );
      this.minimumWidthProperty = new TinyProperty( null );

      // @public {boolean} - Flag for detection of the feature
      this.widthSizable = true;
    }

    /**
     * @public
     *
     * @returns {number|null}
     */
    get preferredWidth() {
      return this.preferredWidthProperty.value;
    }

    /**
     * @public
     *
     * @param {number|null} value
     */
    set preferredWidth( value ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ),
        'preferredWidth should be null or a non-negative finite number' );

      this.preferredWidthProperty.value = value;
    }

    /**
     * @public
     *
     * @returns {number|null}
     */
    get minimumWidth() {
      return this.minimumWidthProperty.value;
    }

    /**
     * @public
     *
     * @param {number|null} value
     */
    set minimumWidth( value ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      this.minimumWidthProperty.value = value;
    }
  };

  // If we're extending into a Node type, include option keys
  // TODO: This is ugly, we'll need to mutate after construction, no?
  if ( type.prototype._mutatorKeys ) {
    clazz.prototype._mutatorKeys = type.prototype._mutatorKeys.concat( WIDTH_SIZABLE_OPTION_KEYS );
  }

  return clazz;
} );

scenery.register( 'WidthSizable', WidthSizable );
export default WidthSizable;