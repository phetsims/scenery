// Copyright 2016, University of Colorado Boulder

/**
 * phet-io handling for PressListener.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
define( function( require ) {
  'use strict';

  // modules
  var assertInstanceOf = require( 'ifphetio!PHET_IO/assertInstanceOf' );
  var phetioInherit = require( 'ifphetio!PHET_IO/phetioInherit' );
  var scenery = require( 'SCENERY/scenery' );
  var ObjectIO = require( 'ifphetio!PHET_IO/types/ObjectIO' );

  /**
   * @param {PressListener} pressListener
   * @param {string} phetioID
   * @constructor
   */
  function TPressListener( pressListener, phetioID ) {
    assert && assertInstanceOf( pressListener, phet.scenery.PressListener );
    ObjectIO.call( this, pressListener, phetioID );
  }

  phetioInherit( ObjectIO, 'TPressListener', TPressListener, {}, {
    documentation: 'Input listener for something that can be pressed.',
    events: [ 'press', 'drag', 'release' ]
  } );

  scenery.register( 'TPressListener', TPressListener );

  return TPressListener;
} );

