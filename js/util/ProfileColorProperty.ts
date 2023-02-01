// Copyright 2021-2023, University of Colorado Boulder

/**
 * ProfileColorProperty is a ColorProperty that changes its value based on the value of colorProfileProperty.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
import arrayRemove from '../../../phet-core/js/arrayRemove.js';
import Namespace from '../../../phet-core/js/Namespace.js';
import Tandem from '../../../tandem/js/Tandem.js';
import { PropertyOptions } from '../../../axon/js/Property.js';
import { Color, colorProfileProperty, ColorProperty, scenery, SceneryConstants } from '../imports.js';
import optionize, { EmptySelfOptions } from '../../../phet-core/js/optionize.js';

// constant
const NAME_SEPARATOR = '.';

// static instances are tracked for iframe communication with the HTML color editor
const instances: ProfileColorProperty[] = [];

type ColorProfileMap = Record<string, Color | string>;

export default class ProfileColorProperty extends ColorProperty {

  // values are mutated by the HTML color wrapper.
  // (scenery-internal)
  public readonly colorProfileMap: ColorProfileMap;

  // Treat as private
  public readonly name: string;

  // Typically use this suffix, but see TANDEM_NAME_SUFFIXES for other possibilities.
  public static override readonly TANDEM_NAME_SUFFIX = 'ColorProperty';

  // See https://github.com/phetsims/scenery/issues/1512
  public static readonly TANDEM_NAME_SUFFIXES = [ ProfileColorProperty.TANDEM_NAME_SUFFIX, 'FillProperty', 'StrokeProperty' ];

  /**
   * @param namespace - namespace that this color belongs to
   * @param colorName - name of the color, unique within namespace
   * @param colorProfileMap - object literal that maps keys (profile names) to ColorDef (that should be immutable)
   * @param [providedOptions]
   */
  public constructor( namespace: Namespace, colorName: string, colorProfileMap: ColorProfileMap, providedOptions?: PropertyOptions<Color> ) {

    const options = optionize<PropertyOptions<Color>, EmptySelfOptions, PropertyOptions<Color>>()( {
      tandem: Tandem.OPTIONAL,

      // So that notifications won't occur when we change from different objects representing the same color.
      // We should never be mutating the Color objects used for ProfileColorProperty.
      valueComparisonStrategy: 'equalsFunction'
    }, providedOptions );

    const tandem = options.tandem;

    // All values are eagerly coerced to Color instances for efficiency (so it only has to be done once) and simplicity
    // (so the types are uniform)
    colorProfileMap = _.mapValues( colorProfileMap, color => {
      // Force Color values to be immutable.
      return Color.toColor( color ).setImmutable();
    } );

    assert && assert( colorProfileMap.hasOwnProperty( SceneryConstants.DEFAULT_COLOR_PROFILE ), 'default color profile must be provided' );
    assert && assert( !!colorProfileMap[ SceneryConstants.DEFAULT_COLOR_PROFILE ], 'default color profile must be truthy' );

    // Fallback to default if a color was not supplied.
    super( Color.toColor( colorProfileMap[ colorProfileProperty.value ] || colorProfileMap[ SceneryConstants.DEFAULT_COLOR_PROFILE ] ), options );

    // See https://github.com/phetsims/scenery/issues/1512
    assert && assert( !this.isPhetioInstrumented() ||
                      _.find( ProfileColorProperty.TANDEM_NAME_SUFFIXES, suffix => tandem.name.endsWith( suffix ) ) ||
                      tandem.name === 'colorProperty' || tandem.name === 'fillProperty' || tandem.name === 'strokeProperty',
      `invalid tandem.name: ${tandem.phetioID}` );

    this.colorProfileMap = colorProfileMap;

    // When the color profile name changes, select the corresponding color.
    colorProfileProperty.link( colorProfileName => {

      // fallback to default if a color not supplied
      this.value = Color.toColor( this.colorProfileMap[ colorProfileName ] || this.colorProfileMap[ SceneryConstants.DEFAULT_COLOR_PROFILE ] );
    } );

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
      const matches = instances.filter( e => e.name === this.name );
      assert && assert( matches.length === 0, 'cannot use the same name for two different ProfileColorProperty instances: ' + name );
    }

    // Register with the static list for the HTML color editor
    instances.push( this );
  }

  public override dispose(): void {
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
      const instanceProperty = instances[ i ];
      if ( instanceProperty.name === data.name ) {
        instanceProperty.colorProfileMap[ colorProfileProperty.value ] = new Color( data.value ).withAlpha( data.alpha );
        instanceProperty.value = Color.toColor( instanceProperty.colorProfileMap[ colorProfileProperty.value ] );
      }
    }
  }
} );

scenery.register( 'ProfileColorProperty', ProfileColorProperty );
