// Copyright 2021, University of Colorado Boulder

/**
 * Singleton Property<string> which chooses between the available color profiles of a simulation, such as 'default', 'project', 'basics', etc.
 *
 * The color profile names available to a simulation are specified in package.json under phet.colorProfiles (or, if not
 * specified, defaults to [ "default" ].  The first listed color profile is one that appears in the sim
 * on startup, unless overridden by the sim or a query parameter.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
import StringProperty from '../../../axon/js/StringProperty.js';
import Tandem from '../../../tandem/js/Tandem.js';
import scenery from '../scenery.js';

// Use the color profile specified in query parameters, or default to 'default'
const initialProfileName = _.hasIn( window, 'phet.chipper.queryParameters.colorProfile' ) ?
                           phet.chipper.queryParameters.colorProfile :
                           'default';

// List of all supported colorProfiles for this simulation
const colorProfiles = _.hasIn( window, 'phet.chipper.colorProfiles' ) ? phet.chipper.colorProfiles : [ 'default' ];

// @public {Property.<string>}
// The current profile name. Change this Property's value to change which profile is currently active.
const colorProfileProperty = new StringProperty( initialProfileName, {

  // TODO: Should we move global.view.colorProfile.profileNameProperty  to global.view.colorProfileProperty ? https://github.com/phetsims/scenery-phet/issues/515
  tandem: Tandem.GLOBAL_VIEW.createTandem( 'colorProfile' ).createTandem( 'profileNameProperty' ),
  validValues: colorProfiles
} );

scenery.register( 'colorProfileProperty', colorProfileProperty );

export default colorProfileProperty;