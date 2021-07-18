// Copyright 2021, University of Colorado Boulder

/**
 * ColorProperty that makes it easy to select a different Color based on the value of the colorProfileProperty.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
import arrayRemove from '../../../phet-core/js/arrayRemove.js';
import scenery from '../scenery.js';
import Color from '../util/Color.js';
import ColorProperty from '../util/ColorProperty.js';
import colorProfileProperty from './colorProfileProperty.js';

// static instances are tracked for iframe communication with the HTML color editor
const instances = [];

class ProfileColorProperty extends ColorProperty {

  /**
   * @param {string} name - name that appears in the HTML color editor
   * @param {Object} colorProfileMap - object literal that maps keys (profile names) to ColorDef
   * @param {Object} [options]
   */
  constructor( name, colorProfileMap, options ) {

    assert && assert( !!name, 'ProfileColorProperty.options.name is required' );

    // All values are eagerly coerced to Color instances for efficiency (so it only has to be done once) and simplicity
    // (so the types are uniform)
    colorProfileMap = _.mapValues( colorProfileMap, Color.toColor );
    super( colorProfileMap[ colorProfileProperty.value ], options );

    // @protected - used elsewhere in this file but outside of this class.
    // values are mutated by the color wrapper.
    this.colorProfileMap = colorProfileMap;

    // When the color profile name changes, select the corresponding color.
    colorProfileProperty.link( colorProfileName => {
      this.value = this.colorProfileMap[ colorProfileName ];
    } );

    // @public (read-only)
    this.name = name;

    this.link( color => {
      if ( window.parent !== window ) {
        window.parent.postMessage( JSON.stringify( {
          type: 'reportColor',
          name: this.name,
          value: color.toHexString()
        } ), '*' );
      }
    } );

    // assert that names are unique
    if ( assert ) {
      const matches = instances.filter( e => e.name === name );
      assert && assert( matches.length === 0, 'cannot use the same name for two different ProfileColorProperty instances: ' + name );
    }

    // Register with the static list for the HTML color editor
    instances.push( this );
  }

  // @public
  dispose() {
    arrayRemove( instances, this );
    super.dispose();
  }
}

// receives iframe communication to set a color
window.addEventListener( 'message', event => {
  let data;
  try {
    data = JSON.parse( event.data );
  }
  catch( e ) {
    // We don't do anything with the caught value. If this happens, it is not JSON. This can happen with the
    // LoL wrappers, see https://github.com/phetsims/joist/issues/484.
  }

  if ( data && data.type === 'setColor' ) {
    for ( let i = 0; i < instances.length; i++ ) {
      const instance = instances[ i ];
      if ( instance.name === data.name ) {
        instance.colorProfileMap[ colorProfileProperty.value ] = new Color( data.value );
        instance.value = instance.colorProfileMap[ colorProfileProperty.value ];
      }
    }
  }
} );

scenery.register( 'ProfileColorProperty', ProfileColorProperty );

export default ProfileColorProperty;