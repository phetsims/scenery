// Copyright 2022, University of Colorado Boulder

/**
 * Whether links should be openable
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
import BooleanProperty from '../../../axon/js/BooleanProperty.js';
import Tandem from '../../../tandem/js/Tandem.js';
import { scenery } from '../imports.js';

const allowLinksProperty = new BooleanProperty( !( window?.phet?.chipper?.queryParameters ) || ( window?.phet?.chipper?.queryParameters?.allowLinks ), {
  tandem: Tandem.GENERAL_MODEL.createTandem( 'allowLinksProperty' )
} );

scenery.register( 'allowLinksProperty', allowLinksProperty );

export default allowLinksProperty;
