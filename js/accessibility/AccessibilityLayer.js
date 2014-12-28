//  Copyright 2002-2014, University of Colorado Boulder

/**
 * The AccessibilityLayer contains any focus highlights that are shown *outside* of particular nodes, such as
 * a focus rectangle, or graphical cursors.  Examples of highlights shown *within* nodes could be highlighting changes
 * or animation.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var Node = require( 'SCENERY/nodes/Node' );
  var Rectangle = require( 'SCENERY/nodes/Rectangle' );
  var Display = require( 'SCENERY/display/Display' );
  var Path = require( 'SCENERY/nodes/Path' );
  var Shape = require( 'KITE/Shape' );

  // constants
  var cursorWidth = 20;

  /**
   * @constructor
   */
  function AccessibilityLayer() {
    var accessibilityLayer = this;
    this.focusRectangle = new Rectangle( 0, 0, 100, 100, 10, 10, {

      stroke: 'blue',

      lineWidth: 2
    } );
    this.focusTriangle = new Path( new Shape().moveTo( 0, 0 ).lineTo( cursorWidth, 0 ).lineTo( cursorWidth / 2, cursorWidth / 10 * 8 ).close(), {fill: 'blue', stroke: 'black', lineWidth: 1} );

    Node.call( this, {children: [this.focusRectangle, this.focusTriangle]} );

    var expand = 5;

    Display.focusedInstanceProperty.link( function( focusedInstance, previousFocusedInstance ) {

      // Animate the focus to a new node
      if ( focusedInstance && previousFocusedInstance ) {
        var node = focusedInstance.node;

        var bounds = node.getGlobalBounds();

        accessibilityLayer.focusRectangle.visible = true;
        accessibilityLayer.focusTriangle.visible = true;

        var targetBounds = {
          x: bounds.left - expand,
          y: bounds.top - expand,
          width: bounds.width + expand * 2,
          height: bounds.height + expand * 2
        };

        // For accessibility animation, scenery requires the TWEEN.js library
        new TWEEN.Tween( {
          x: accessibilityLayer.focusRectangle.getRectX(),
          y: accessibilityLayer.focusRectangle.getRectY(),
          width: accessibilityLayer.focusRectangle.getRectWidth(),
          height: accessibilityLayer.focusRectangle.getRectHeight()
        } ).to( targetBounds, 300 ).
          easing( TWEEN.Easing.Cubic.InOut ).
          onUpdate( function() {
            accessibilityLayer.focusRectangle.setRect( this.x, this.y, this.width, this.height, 10, 10 );
            accessibilityLayer.focusTriangle.bottom = this.y;
            accessibilityLayer.focusTriangle.centerX = this.x + this.width / 2;
          } ).
          start();
      }

      // Show the focus, when there was no focus node before.
      else if ( focusedInstance ) {

        var b = focusedInstance.node.getGlobalBounds();

        var newTargetBounds = {
          x: b.left - expand,
          y: b.top - expand,
          width: b.width + expand * 2,
          height: b.height + expand * 2
        };

        accessibilityLayer.focusRectangle.visible = true;
        accessibilityLayer.focusTriangle.visible = true;
        accessibilityLayer.focusRectangle.setRect( newTargetBounds.x, newTargetBounds.y, newTargetBounds.width, newTargetBounds.height, 10, 10 );

        accessibilityLayer.focusTriangle.bottom = newTargetBounds.y;
        accessibilityLayer.focusTriangle.centerX = newTargetBounds.x + newTargetBounds.width / 2;
      }

      // No focused node.
      else {
        accessibilityLayer.focusRectangle.visible = false;
        accessibilityLayer.focusTriangle.visible = false;
      }
    } );
  }

  return inherit( Node, AccessibilityLayer );
} );