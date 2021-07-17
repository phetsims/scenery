// Copyright 2021, University of Colorado Boulder

/**
 * Color property that takes an object literal where the keys are profile names and the values are ColorDef.
 * @author Sam Reid
 */
import arrayRemove from '../../../phet-core/js/arrayRemove.js';
import scenery from '../scenery.js';
import Color from '../util/Color.js';
import ColorProperty from '../util/ColorProperty.js';
import colorProfileNameProperty from './colorProfileNameProperty.js';

// static instances are tracked for iframe communication with the HTML color editor
const instances = [];

// TODO https://github.com/phetsims/scenery-phet/issues/515 rename to ProfileColorProperty
class ColorProfileProperty extends ColorProperty {

  /**
   * @param {string} name - name that appears in the HTML color editor
   * @param {Object|ColorDef} colorProfileMap - object literal that maps keys (profile names) to ColorDef, or just a default colorDef
   * @param {Object} [options]
   */
  constructor( name, colorProfileMap, options ) {

    assert && assert( name, 'ColorProfileProperty.options.name is required' );

    // All values are eagerly coerced to Color instances for efficiency (so it only has to be done once) and simplicity
    // (so the types are uniform)
    colorProfileMap = _.mapValues( colorProfileMap, Color.toColor );
    super( colorProfileMap[ colorProfileNameProperty.value ], options );

    // @protected - used elsewhere in this file but outside of this class.
    // values are mutated by the color wrapper.
    this.colorProfileMap = colorProfileMap;

    // When the color profile name changes, select the corresponding color.
    colorProfileNameProperty.link( colorProfileName => {
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
      assert && assert( matches.length === 0, 'cannot use the same name for two different ColorProfileProperty instances: ' + name );
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
        instance.colorProfileMap[ colorProfileNameProperty.value ] = new Color( data.value );
        instance.value = instance.colorProfileMap[ colorProfileNameProperty.value ];
      }
    }
  }
} );

scenery.register( 'ColorProfileProperty', ColorProfileProperty );

export default ColorProfileProperty;