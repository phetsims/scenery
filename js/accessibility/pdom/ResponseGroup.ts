// Copyright 2026, University of Colorado Boulder

/**
 * Named response groups for accessible responses. These are values to use with responseGroup option
 * of DescriptionResponseOptions when using addAccessibleContextResponse or any related methods.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import scenery from '../../scenery.js';

class ResponseGroup {

  // A general response group for user interface components. So that rapid interaction with a user interface
  // component does not cause a flood of responses, but the group does not interfere or get inerrupted by
  // other information.
  public static readonly USER_INTERFACE = 'user-interface';
}

scenery.register( 'ResponseGroup', ResponseGroup );

export default ResponseGroup;
