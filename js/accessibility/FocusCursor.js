//  Copyright 2002-2014, University of Colorado Boulder

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

    focusedBoundsProperty.link( function( targetBounds, previousBounds ) {
      focusCursor.visible = (targetBounds !== null);
      if ( targetBounds && previousBounds ) {
        // For accessibility animation, scenery requires the TWEEN.js library
        new TWEEN.Tween( {
          x: previousBounds.x,
          y: previousBounds.y,
          width: previousBounds.width,
          height: previousBounds.height
        } ).to( targetBounds, 300 ).
          easing( TWEEN.Easing.Cubic.InOut ).
          onUpdate( function() {
            focusCursor.bottom = this.y;
            focusCursor.centerX = this.x + this.width / 2;
          } ).
          start();
      }
      else if ( targetBounds && previousBounds === null ) {
        focusCursor.bottom = targetBounds.y;
        focusCursor.centerX = targetBounds.x + targetBounds.width / 2;
      }
      else {
        //should be invisible, nothing else to do here
      }
    } );

    Property.multilink( [ focusedBoundsProperty, focusIndicatorProperty ], function( focusedBounds, focusIndicator ) {
      var visible = focusedBounds !== null && focusIndicator === 'cursor';
      focusCursor.visible = visible;
    } );
  }

  return inherit( Path, FocusCursor );
} );