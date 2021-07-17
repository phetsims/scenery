// Copyright 2021, University of Colorado Boulder

import StringProperty from '../../../axon/js/StringProperty.js';
import Tandem from '../../../tandem/js/Tandem.js';
import scenery from '../scenery.js';

// TODO: https://github.com/phetsims/scenery-phet/issues/515 Documentation
const initialProfileName = _.hasIn( window, 'phet.chipper.queryParameters.colorProfile' ) ?
                           phet.chipper.queryParameters.colorProfile :
                           'default';
const colorProfiles = _.hasIn( window, 'phet.chipper.colorProfiles' ) ?
                      phet.chipper.colorProfiles : [ 'default' ];

// @public {Property.<string>}
// The current profile name. Change this Property's value to change which profile is currently active.
const colorProfileNameProperty = new StringProperty( initialProfileName, {

  // TODO: Should we move global.view.colorProfile.profileNameProperty  to global.view.colorProfileNameProperty ? https://github.com/phetsims/scenery-phet/issues/515
  tandem: Tandem.GLOBAL_VIEW.createTandem( 'colorProfile' ).createTandem( 'profileNameProperty' ),
  validValues: colorProfiles
} );

scenery.register( 'colorProfileNameProperty', colorProfileNameProperty );

export default colorProfileNameProperty;