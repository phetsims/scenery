// Copyright 2015-2016, University of Colorado Boulder

/**
 * Focus highlight overlay for accessible displays.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );

  var Color = require( 'SCENERY/util/Color' );
  var Node = require( 'SCENERY/nodes/Node' );
  var Path = require( 'SCENERY/nodes/Path' );
  var Rectangle = require( 'SCENERY/nodes/Rectangle' );
  var scenery = require( 'SCENERY/scenery' );
  var Shape = require( 'KITE/Shape' );
  var TransformTracker = require( 'SCENERY/util/TransformTracker' );
  var Vector2 = require( 'DOT/Vector2' );

 // determined by inspection, base widths of focus highlight, transform of shape/bounds will change highlight line width
  var INNER_LINE_WIDTH_BASE = 2.5;
  var OUTER_LINE_WIDTH_BASE = 4;

  /**
   * @constructor
   *
   * @param {Display} display
   * @param {Node} focusRootNode - the root node of our display
   */
  function FocusOverlay( display, focusRootNode ) {
    this.display = display; // @private {Display}
    this.focusRootNode = focusRootNode; // @private {Node} - The root Node of our child display

    // When the focus changes, all of these are modified.
    this.trail = null; // @private {Trail|null}
    this.node = null; // @private {Node|null}
    this.mode = null; // @private {String|null} - defaults to bounds but can be overwritten by the node's focus highlight
    this.transformTracker = null; // @private {TransformTracker|null}

    // @private {boolean} - If true, the next update() will trigger an update to the highlight's transform.
    this.transformDirty = true;

    // @private - The main node for the highlight. It will be transformed.
    this.highlightNode = new Node();
    this.focusRootNode.addChild( this.highlightNode );

    // @private {Display}
    this.focusDisplay = new scenery.Display( this.focusRootNode, {
      width: this.width,
      height: this.height,
      allowWebGL: display._allowWebGL,
      allowCSSHacks: false,
      accessibility: false,
      isApplication: false,
      interactive: false
    } );

    // @private {HTMLElement}
    this.domElement = this.focusDisplay.domElement;
    this.domElement.style.pointerEvents = 'none';

    // @private Bounds highlight
    this.boundsHighlight = new Rectangle( 0, 0, 0, 0, { stroke: FocusOverlay.focusColor, visible: false } );
    this.innerBoundsHighlight = new Rectangle( 0, 0, 0, 0, { stroke: FocusOverlay.innerFocusColor } );
    this.boundsHighlight.addChild( this.innerBoundsHighlight );

    // @private Shape highlight
    this.shapeHighlight = new Path( null, { stroke: FocusOverlay.focusColor, visible: false } );
    this.innerShapeHighlight = new Path( null, { stroke: FocusOverlay.innerFocusColor } );
    this.shapeHighlight.addChild( this.innerShapeHighlight );

    // @private Node highlight
    this.nodeHighlight = null;

    this.highlightNode.addChild( this.boundsHighlight );
    this.highlightNode.addChild( this.shapeHighlight );

    // @private - Listeners bound once, so we can access them for removal.
    this.boundsListener = this.onBoundsChange.bind( this );
    this.transformListener = this.onTransformChange.bind( this );
    this.focusListener = this.onFocusChange.bind( this );

    scenery.Display.focusProperty.link( this.focusListener );
  }

  scenery.register( 'FocusOverlay', FocusOverlay );

  inherit( Object, FocusOverlay, {
    dispose: function() {
      if ( this.hasHighlight() ) {
        this.deactivateHighlight();
      }

      scenery.Display.focusProperty.unlink( this.focusListener );
    },

    hasHighlight: function() {
      return !!this.trail;
    },

    /**
     * Activates the highlight, choosing a mode for whether the highlight will be a shape, node, or bounds.
     * @private
     *
     * @param {Trail} trail - The focused trail to highlight. It assumes that this trail is in this display.
     */
    activateHighlight: function( trail ) {
      this.trail = trail;
      this.node = trail.lastNode();
      this.transformTracker = new TransformTracker( trail, {
        isStatic: true
      } );
      this.transformTracker.addListener( this.transformListener );

      // Invisible mode - no focus highlight
      if ( this.node.accessibleContent.focusHighlight === 'invisible' ) {
        this.mode = 'invisible';
      }
      // Shape mode
      else if ( this.node.accessibleContent.focusHighlight instanceof Shape ) {
        this.mode = 'shape';

        this.shapeHighlight.visible = true;
        this.shapeHighlight.shape = this.innerShapeHighlight.shape = this.node.accessibleContent.focusHighlight;
      }
      // Node mode
      else if ( this.node.accessibleContent.focusHighlight instanceof Node ) {
        this.mode = 'node';

        // If focusHighlightLayerable, then the focusHighlight is just a node in the scene graph, so set it visible
        if ( this.node.accessibleContent.focusHighlightLayerable ) {
          this.node.accessibleContent.focusHighlight.visible = true;
        }
        else {
          this.nodeHighlight = this.node.accessibleContent.focusHighlight;

          // Use the node itself as the highlight
          this.highlightNode.addChild( this.nodeHighlight );
        }
      }
      // Bounds mode
      else {
        this.mode = 'bounds';

        this.boundsHighlight.visible = true;
        this.node.onStatic( 'localBounds', this.boundsListener );

        this.onBoundsChange();
      }

      this.transformDirty = true;
    },

    /**
     * Deactivates the current highlight, disposing and removing listeners as necessary.
     * @private
     */
    deactivateHighlight: function() {
      if ( this.mode === 'shape' ) {
        this.shapeHighlight.visible = false;
      }
      else if ( this.mode === 'node' ) {

        // If focusHighlightLayerable, then the focusHighlight is just a node in the scene graph, so set it invisible
        if ( this.node.accessibleContent.focusHighlightLayerable ) {
          this.node.accessibleContent.focusHighlight.visible = false;
        }
        else {
          this.highlightNode.removeChild( this.nodeHighlight );
          this.nodeHighlight = null;
        }
      }
      else if ( this.mode === 'bounds' ) {
        this.boundsHighlight.visible = false;
        this.node.offStatic( 'localBounds', this.boundsListener );
      }

      this.trail = null;
      this.node = null;
      this.mode = null;
      this.transformTracker.removeListener( this.transformListener );
      this.transformTracker.dispose();
    },

    // Called from FocusOverlay after transforming the highlight. Only called when the transform changes.
    afterTransform: function() {
      if ( this.mode === 'shape' ) {
        var unitShapeWidthMagnitude = this.shapeHighlight.transform.transformDelta2( Vector2.X_UNIT ).magnitude();
        this.shapeHighlight.lineWidth = OUTER_LINE_WIDTH_BASE / unitShapeWidthMagnitude;
        this.innerShapeHighlight.lineWidth = INNER_LINE_WIDTH_BASE / unitShapeWidthMagnitude;
      }
      else if ( this.mode === 'bounds' ) {
        var unitBoundsWidthMagnitude = this.boundsHighlight.transform.transformDelta2( Vector2.X_UNIT ).magnitude();
        this.boundsHighlight.lineWidth = OUTER_LINE_WIDTH_BASE / unitBoundsWidthMagnitude;
        this.innerBoundsHighlight.lineWidth = INNER_LINE_WIDTH_BASE / unitBoundsWidthMagnitude;
      }
    },

    onTransformChange: function() {
      this.transformDirty = true;
    },

    // Called when bounds change on our node when we are in "Bounds" mode
    onBoundsChange: function() {
      this.boundsHighlight.setRectBounds( FocusOverlay.getStaticFocusHighlightBounds( this.node ) );
      this.innerBoundsHighlight.setRectBounds( FocusOverlay.getStaticFocusHighlightBounds( this.node ) );
    },

    // Called when the main Scenery focus pair (Display,Trail) changes.
    onFocusChange: function( focus ) {
      var newTrail = ( focus && focus.display === this.display ) ? focus.trail : null;

      if ( this.hasHighlight() ) {
        this.deactivateHighlight();
      }

      if ( newTrail ) {
        this.activateHighlight( newTrail );
      }
    },

    update: function() {
      // Transform the highlight to match the position of the node
      if ( this.hasHighlight() && this.transformDirty ) {
        this.transformDirty = false;

        this.highlightNode.setMatrix( this.transformTracker.matrix );
        this.afterTransform();
      }

      if ( !this.display.size.equals( this.focusDisplay.size ) ) {
        this.focusDisplay.setWidthHeight( this.display.width, this.display.height );
      }
      this.focusDisplay.updateDisplay();
    }
  }, {
    focusColor: new Color( 'rgba(212,19,106,0.5)' ),
    innerFocusColor: new Color( 'rgba(250,40,135,0.9)' ),

    /**
     * Get the bounds of a focus highlight for a node. The space between the inner most edge of the focus highlight
     * and the bounds of the node will be 1 / 2 times the line width of the focus highlight. Will provide
     * consistent spacing between node and inner edge of focus highlight, see
     * see https://github.com/phetsims/scenery/issues/677
     *
     * @param {Node} node - the node that will receive the focus highlight
     * @return {Bounds2}
     */
    getStaticFocusHighlightBounds: function( node ) {
      var unitBoundsWidthMagnitude = node.transform.transformDelta2( Vector2.X_UNIT ).magnitude();
      var outerHalfWidth = ( OUTER_LINE_WIDTH_BASE / unitBoundsWidthMagnitude ) * ( 3 / 4 );
      return node.localBounds.dilated( outerHalfWidth );
    } 
  } );

  return FocusOverlay;
} );
