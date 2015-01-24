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
  var Input = require( 'SCENERY/input/Input' );
  var FocusRectangle = require( 'SCENERY/accessibility/FocusRectangle' );
  var FocusCursor = require( 'SCENERY/accessibility/FocusCursor' );
  var DerivedProperty = require( 'AXON/DerivedProperty' );

  /**
   * @constructor
   */
  function AccessibilityLayer() {

    var expand = 5;
    var focusedBoundsProperty = new DerivedProperty( [ Input.focusedInstanceProperty ], function( focusedInstance ) {
      if ( focusedInstance ) {
        var b = focusedInstance.node.getGlobalBounds();

        // TODO: A real live dot.Rectangle
        return {
          x:      b.left - expand,
          y:      b.top - expand,
          width:  b.width + expand * 2,
          height: b.height + expand * 2
        };
      }
      else {
        return null;
      }
    } );

    var focusIndicatorProperty = new DerivedProperty( [ Input.focusedInstanceProperty ], function( focusedInstance ) {
      if ( focusedInstance ) {
        return focusedInstance.node.focusIndicator || 'rectangle';
      }
      else {
        return null;
      }
    } );

    this.focusRectangle = new FocusRectangle( focusedBoundsProperty, focusIndicatorProperty );
    this.focusCursor = new FocusCursor( focusedBoundsProperty, focusIndicatorProperty );

    Node.call( this, { children: [ this.focusRectangle, this.focusCursor ] } );
  }

  return inherit( Node, AccessibilityLayer );
} );