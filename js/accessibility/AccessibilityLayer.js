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

  /**
   * @constructor
   */
  function AccessibilityLayer() {
    var accessibilityLayer = this;
    this.focusRectangle = new Rectangle( 0, 0, 100, 100, 10, 10, {

      stroke: 'blue',

      lineWidth: 2
    } );

    Node.call( this, {children: [this.focusRectangle]} );

    var expand = 5;

    Display.focusedInstanceProperty.link( function( focusedInstance, previousFocusedInstance ) {

      // Animate the focus to a new node
      if ( focusedInstance && previousFocusedInstance ) {
        var node = focusedInstance.node;

        accessibilityLayer.focusRectangle.visible = true;

        // For accessibility animation, scenery requires the TWEEN.js library
        new TWEEN.Tween( {
          x: accessibilityLayer.focusRectangle.getRectX(),
          y: accessibilityLayer.focusRectangle.getRectY(),
          width: accessibilityLayer.focusRectangle.getRectWidth(),
          height: accessibilityLayer.focusRectangle.getRectHeight()
        } ).to( {
            x: node.left - expand,
            y: node.top - expand,
            width: node.width + expand * 2,
            height: node.height + expand * 2
          }, 300 ).
          easing( TWEEN.Easing.Cubic.InOut ).
          onUpdate( function() {
            accessibilityLayer.focusRectangle.setRect( this.x, this.y, this.width, this.height, 10, 10 );
          } ).
          start();
      }

      // Show the focus, when there was no focus node before.
      else if ( focusedInstance ) {
        accessibilityLayer.focusRectangle.visible = true;
        accessibilityLayer.focusRectangle.setRect( focusedInstance.node.left - expand, focusedInstance.node.top - expand, focusedInstance.node.width + expand * 2, focusedInstance.node.height + expand * 2, 10, 10 );
      }

      // No focused node.
      else {
        accessibilityLayer.focusRectangle.visible = false;
      }
    } );
  }

  return inherit( Node, AccessibilityLayer );
} );