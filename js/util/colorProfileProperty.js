// Copyright 2021-2022, University of Colorado Boulder

/**
 * Singleton Property<string> which chooses between the available color profiles of a simulation, such as 'default',
 * 'project', 'basics', etc.
 *
 * The color profile names available to a simulation are specified in package.json under phet.colorProfiles (or, if not
 * specified, defaults to [ "default" ].  The first color profile that is listed will appear in the sim on startup,
 * unless overridden by the sim or the colorProfile query parameter.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
import StringProperty from '../../../axon/js/StringProperty.js';
import Tandem from '../../../tandem/js/Tandem.js';
import { scenery, SceneryConstants } from '../imports.js';

// Use the color profile specified in query parameters, or default to 'default'
const initialProfileName = _.hasIn( window, 'phet.chipper.queryParameters.colorProfile' ) ?
                           phet.chipper.queryParameters.colorProfile :
                           SceneryConstants.DEFAULT_COLOR_PROFILE;

// List of all supported colorProfiles for this simulation
const colorProfiles = _.hasIn( window, 'phet.chipper.colorProfiles' ) ? phet.chipper.colorProfiles : [ SceneryConstants.DEFAULT_COLOR_PROFILE ];

// @public {Property.<string>}
// The current profile name. Change this Property's value to change which profile is currently active.
const colorProfileProperty = new StringProperty( initialProfileName, {
  tandem: Tandem.GENERAL_VIEW.createTandem( 'colorProfileProperty' ),
  phetioFeatured: true,
  validValues: colorProfiles
} );

scenery.register( 'colorProfileProperty', colorProfileProperty );

export default colorProfileProperty;