// Copyright 2002-2014, University of Colorado Boulder

/**
 *
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var Path = require( 'SCENERY/nodes/Path' );
  var Shape = require( 'KITE/Shape' );
  var Property = require( 'AXON/Property' );

  // constants
  var cursorWidth = 20;

  /**
   *
   * @constructor
   */
  function FocusCursor( focusedBoundsProperty, focusIndicatorProperty ) {
    var focusCursor = this;

    Path.call( this, new Shape().moveTo( 0, 0 ).lineTo( cursorWidth, 0 ).lineTo( cursorWidth / 2, cursorWidth / 10 * 8 ).close(), {
      fill: 'blue',
      stroke: 'black',
      lineWidth: 1
    } );

    // TODO: Don't update when invisible
    focusedBoundsProperty.link( function( focusedBounds ) {
      if ( focusedBounds ) {
        focusCursor.bottom = focusedBounds.y;
        focusCursor.centerX = focusedBounds.x + focusedBounds.width / 2;
      }
    } );

    Property.multilink( [ focusedBoundsProperty, focusIndicatorProperty ], function( focusedBounds, focusIndicator ) {
      var visible = focusedBounds !== null && focusIndicator === 'cursor';
      focusCursor.visible = visible;
    } );
  }

  return inherit( Path, FocusCursor );
} );