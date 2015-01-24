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
  var Rectangle = require( 'SCENERY/nodes/Rectangle' );
  var Property = require( 'AXON/Property' );

  /**
   *
   * @constructor
   */
  function FocusRectangle( focusedBoundsProperty, focusIndicatorProperty ) {
    var focusRectangle = this;

    Rectangle.call( this, 0, 0, 100, 100, 10, 10, {
      stroke: 'blue',
      visible: false,
      lineWidth: 2
    } );
    focusedBoundsProperty.link( function( targetBounds, previousBounds ) {
      if ( targetBounds && previousBounds ) {
        // For accessibility animation, scenery requires the TWEEN.js library
        new TWEEN.Tween( {
          x: focusRectangle.getRectX(),
          y: focusRectangle.getRectY(),
          width: focusRectangle.getRectWidth(),
          height: focusRectangle.getRectHeight()
        } ).to( targetBounds, 300 ).
          easing( TWEEN.Easing.Cubic.InOut ).
          onUpdate( function() {
            focusRectangle.setRect( this.x, this.y, this.width, this.height, 10, 10 );
          } ).
          start();
      }
      else if ( targetBounds && previousBounds === null ) {
        focusRectangle.setRect( targetBounds.x, targetBounds.y, targetBounds.width, targetBounds.height, 10, 10 );
      }
      else {
        //should be invisible, nothing else to do here
      }
    } );

    Property.multilink( [ focusedBoundsProperty, focusIndicatorProperty ], function( focusedBounds, focusIndicator ) {
      var visible = focusedBounds !== null && focusIndicator === 'rectangle';
      focusRectangle.visible = visible;
    } );
  }

  return inherit( Rectangle, FocusRectangle );
} );