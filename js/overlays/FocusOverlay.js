// Copyright 2002-2014, University of Colorado Boulder

/**
 * Focus highlight overlay for accessible displays.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );

  var scenery = require( 'SCENERY/scenery' );
  var Vector2 = require( 'DOT/Vector2' );
  var Rectangle = require( 'SCENERY/nodes/Rectangle' );
  var Color = require( 'SCENERY/util/Color' );
  var Shape = require( 'KITE/Shape' );
  var Node = require( 'SCENERY/nodes/Node' );
  var Path = require( 'SCENERY/nodes/Path' );
  // var Display = require( 'SCENERY/display/Display' );

  scenery.FocusOverlay = function FocusOverlay( display, focusRootNode ) {
    var overlay = this;

    this.display = display;
    this.focusRootNode = focusRootNode;

    this.focusDisplay = new scenery.Display( this.focusRootNode, {
      width: this.width,
      height: this.height,
      allowCSSHacks: false,
      accessibility: false,
      isApplication: false,
      interactive: false
    } );

    this.domElement = this.focusDisplay.domElement;

    var boundsHighlight = new Rectangle( 0, 0, 0, 0, { stroke: FocusOverlay.focusColor, visible: false } );
    var innerBoundsHighlight = new Rectangle( 0, 0, 0, 0, { stroke: FocusOverlay.innerFocusColor } );
    boundsHighlight.addChild( innerBoundsHighlight );
    var shapeHighlight = new Path( null, { stroke: FocusOverlay.focusColor, visible: false } );
    var innerShapeHighlight = new Path( null, { stroke: FocusOverlay.innerFocusColor } );
    shapeHighlight.addChild( innerShapeHighlight );
    var nodeHighlight = null;
    this.focusRootNode.addChild( boundsHighlight );
    this.focusRootNode.addChild( shapeHighlight );

    scenery.Display.focusProperty.link( function( focus ) {
      if ( focus && focus.display === display ) {
        var trail = focus.trail;
        var node = trail.lastNode();

        if ( nodeHighlight ) {
          overlay.focusRootNode.removeChild( nodeHighlight );
          nodeHighlight = null;
        }

        if ( node.accessibleContent.focusHighlight instanceof Shape ) {
          shapeHighlight.visible = true;
          boundsHighlight.visible = false;

          shapeHighlight.shape = innerShapeHighlight.shape = node.accessibleContent.focusHighlight;
          shapeHighlight.setMatrix( focus.trail.getMatrix() );
          shapeHighlight.lineWidth = 4 / shapeHighlight.transform.transformDelta2( Vector2.X_UNIT ).magnitude();
          innerShapeHighlight.lineWidth = 2.5 / shapeHighlight.transform.transformDelta2( Vector2.X_UNIT ).magnitude();
        }
        else if ( node.accessibleContent.focusHighlight instanceof Node ) {
          nodeHighlight = node.accessibleContent.focusHighlight;
          overlay.focusRootNode.addChild( nodeHighlight );
          nodeHighlight.setMatrix( focus.trail.getMatrix() );
        }
        else {
          shapeHighlight.visible = false;
          boundsHighlight.visible = true;

          boundsHighlight.setRectBounds( node.localBounds );
          innerBoundsHighlight.setRectBounds( node.localBounds );
          boundsHighlight.setMatrix( focus.trail.getMatrix() );
          boundsHighlight.lineWidth = 4 / boundsHighlight.transform.transformDelta2( Vector2.X_UNIT ).magnitude();
          innerBoundsHighlight.lineWidth = 2.5 / boundsHighlight.transform.transformDelta2( Vector2.X_UNIT ).magnitude();
        }
      }
      else {
        boundsHighlight.visible = false;
        shapeHighlight.visible = false;
      }

    } );
  };
  var FocusOverlay = scenery.FocusOverlay;

  inherit( Object, FocusOverlay, {
    dispose: function() {

    },

    update: function() {
      this.focusDisplay.updateDisplay();
      if ( !this.display.size.equals( this.focusDisplay.size ) ) {
        this.focusDisplay.setWidthHeight( this.display.width, this.display.height );
      }
    }
  }, {
    focusColor: new Color( 'rgba(0,100,255,0.5)' ),
    innerFocusColor: new Color( 'rgba(0,200,255,0.9)' )
  } );

  return FocusOverlay;
} );
