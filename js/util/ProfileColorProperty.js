// Copyright 2021, University of Colorado Boulder

/**
 * ProfileColorProperty is a ColorProperty that changes its value based on the value of colorProfileProperty.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
import arrayRemove from '../../../phet-core/js/arrayRemove.js';
import merge from '../../../phet-core/js/merge.js';
import Namespace from '../../../phet-core/js/Namespace.js';
import Tandem from '../../../tandem/js/Tandem.js';
import scenery from '../scenery.js';
import SceneryConstants from '../SceneryConstants.js';
import Color from '../util/Color.js';
import ColorProperty from '../util/ColorProperty.js';
import colorProfileProperty from './colorProfileProperty.js';

// constant
const NAME_SEPARATOR = '.';

// static instances are tracked for iframe communication with the HTML color editor
const instances = [];

class ProfileColorProperty extends ColorProperty {

  /**
   * @param {Namespace} namespace - namespace that this color belongs to
   * @param {string} colorName - name of the color, unique within namespace
   * @param {Object} colorProfileMap - object literal that maps keys (profile names) to ColorDef
   * @param {Object} [options]
   */
  constructor( namespace, colorName, colorProfileMap, options ) {

    assert && assert( namespace instanceof Namespace );
    assert && assert( typeof colorName === 'string' );

    options = merge( {
      tandem: Tandem.OPTIONAL
    }, options );

    // All values are eagerly coerced to Color instances for efficiency (so it only has to be done once) and simplicity
    // (so the types are uniform)
    colorProfileMap = _.mapValues( colorProfileMap, Color.toColor );

    // Fallback to default if a color was not supplied.
    super( colorProfileMap[ colorProfileProperty.value ] || colorProfileMap[ SceneryConstants.DEFAULT_COLOR_PROFILE ], options );

    assert && assert( !this.isPhetioInstrumented() ||
                      options.tandem.name.endsWith( 'ColorProperty' ) ||
                      options.tandem.name === 'colorProperty',
      `Property tandem.name must end with ColorProperty: ${options.tandem.phetioID}` );

    // @public (ProfileColorProperty.js) - values are mutated by the HTML color wrapper.
    this.colorProfileMap = colorProfileMap;

    // When the color profile name changes, select the corresponding color.
    colorProfileProperty.link( colorProfileName => {

      // fallback to default if a color not supplied
      this.value = this.colorProfileMap[ colorProfileName ] || this.colorProfileMap[ SceneryConstants.DEFAULT_COLOR_PROFILE ];
    } );

    // @private to this file (read-only)
    this.name = `${namespace.name}${NAME_SEPARATOR}${colorName}`;

    // On initialization and when the color changes, send a message to the parent frame identifying the color value.
    // The HTML color editor wrapper listens for these messages and displays the color values.
    this.link( color => {
      if ( window.parent !== window ) {
        window.parent.postMessage( JSON.stringify( {
          type: 'reportColor',
          name: this.name,
          value: color.toHexString(),
          alpha: color.getAlpha()
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

// Listen for messages from the HTML color editor wrapper with new color values.
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
        instance.colorProfileMap[ colorProfileProperty.value ] = new Color( data.value ).withAlpha( data.alpha );
        instance.value = instance.colorProfileMap[ colorProfileProperty.value ];
      }
    }
  }
} );

scenery.register( 'ProfileColorProperty', ProfileColorProperty );

export default ProfileColorProperty;