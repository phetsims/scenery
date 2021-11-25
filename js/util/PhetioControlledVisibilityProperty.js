// Copyright 2021, University of Colorado Boulder

/**
 * An enhanced DerivedProperty to be used as a visibilityProperty to a Node when phet-io needs to be able to
 * independently set visibility to false in a way that can't be overridden by the sim.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import BooleanProperty from '../../../axon/js/BooleanProperty.js';
import DerivedProperty from '../../../axon/js/DerivedProperty.js';
import merge from '../../../phet-core/js/merge.js';
import Tandem from '../../../tandem/js/Tandem.js';
import { scenery } from '../imports.js';

class PhetioControlledVisibilityProperty extends DerivedProperty {
  /**
   * @param {Array.<Property<boolean>|TinyProperty<boolean>>} dependencies - Properties that this Property's value is derived from
   * @param {(...x:any[])=>boolean} derivation - function that derives this Property's value, expects args in the same order as dependencies
   * @param {Object} [options] - see Property
   */
  constructor( dependencies, derivation, options ) {

    options = merge( {
      nodeTandem: Tandem.REQUIRED
    }, options );

    // We'll create an instrumented BooleanProperty that, when toggled to false, will hide the node regardless of
    // what the derivation would return.
    const visibleProperty = new BooleanProperty( true, {
      tandem: options.nodeTandem.createTandem( 'visibleProperty' )
    } );

    super( [ visibleProperty, ...dependencies ], ( visible, ...args ) => visible && derivation( ...args ) );
  }
}

scenery.register( 'PhetioControlledVisibilityProperty', PhetioControlledVisibilityProperty );
export default PhetioControlledVisibilityProperty;