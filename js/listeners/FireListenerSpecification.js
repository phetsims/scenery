// Copyright 2020, University of Colorado Boulder

/**
 * Specification for FireListener
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
import Emitter from '../../../axon/js/Emitter.js';
import EmitterSpecification from '../../../axon/js/EmitterSpecification.js';
import merge from '../../../phet-core/js/merge.js';
import EventType from '../../../tandem/js/EventType.js';
import Tandem from '../../../tandem/js/Tandem.js';
import NullableIO from '../../../tandem/js/types/NullableIO.js';
import SceneryEvent from '../input/SceneryEvent.js';
import scenery from '../scenery.js';
import PressListenerSpecification from './PressListenerSpecification.js';

class FireListenerSpecification extends PressListenerSpecification {

  /**
   * @param {Object} [options]
   */
  constructor( options ) {
    options = merge( {
      firedEmitterOptions: {
        phetioType: Emitter.EmitterIO( [ NullableIO( SceneryEvent.SceneryEventIO ) ] ),
        phetioEventType: EventType.USER
      },

      // TODO https://github.com/phetsims/phet-io/issues/1657 how to make sure this is supplied?  Is it as simple as omitting the && options.tandem.supplied check below?
      tandem: Tandem.REQUIRED
    }, options );

    super( options );

    // In PhET-iO, we can look up the concrete emitter.
    // TODO: The firedEmitter is actually private in FireListener--we could make it public for testing? see https://github.com/phetsims/phet-io/issues/1657
    if ( Tandem.VALIDATION && options.tandem.supplied ) {

      // @public (read-only)
      this.firedEmitter = new EmitterSpecification( options.firedEmitterOptions );
    }
  }

  // @public
  test( fireListener ) {
    super.test( fireListener );
    if ( Tandem.VALIDATION ) {
      const firedEmitter = phet.phetio.phetioEngine.getPhetioObject( this.options.tandem.createTandem( 'firedEmitter' ).phetioID );
      this.firedEmitter && this.firedEmitter.test( firedEmitter );
    }
  }
}

scenery.register( 'FireListenerSpecification', FireListenerSpecification );
export default FireListenerSpecification;