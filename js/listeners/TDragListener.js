// Copyright 2016, University of Colorado Boulder

/**
 * phet-io handling for DragListener.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
define( function( require ) {
  'use strict';

  // modules
  var assertInstanceOf = require( 'ifphetio!PHET_IO/assertInstanceOf' );
  var phetioInherit = require( 'ifphetio!PHET_IO/phetioInherit' );
  var scenery = require( 'SCENERY/scenery' );
  var TPressListener = require( 'SCENERY/listeners/TPressListener' );

  /**
   * @param {DragListener} dragListener
   * @param {string} phetioID
   * @constructor
   */
  function TDragListener( dragListener, phetioID ) {
    assert && assertInstanceOf( dragListener, phet.scenery.DragListener );
    TPressListener.call( this, dragListener, phetioID );
  }

  phetioInherit( TPressListener, 'TDragListener', TDragListener, {}, {
    documentation: 'Input listener for something that can be dragged.',
    events: [ 'start', 'end' ]
  } );

  scenery.register( 'TDragListener', TDragListener );

  return TDragListener;
} );

