// Copyright 2002-2012, University of Colorado

/**
 * Basic dragging for a node.
 *
 * TODO: stick node in place if ancestor changes transform (while dragging)
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

var scenery = scenery || {};

(function(){
  "use strict";
  
  /*
   * Allowed options: {
   *    allowTouchSnag: false // allow touch swipes across an object to pick it up,
   *    start: null           // if non-null, called when a drag is started. start( finger, trail, event )
   *    end: null             // if non-null, called when a drag is ended.   end( finger, trail, event )
   * }
   */
  scenery.SimpleDragHandler = function( options ) {
    var handler = this;
    
    this.options = options;
    
    this.dragging              = false;     // whether a node is being dragged with this handler
    this.finger                = null;      // the finger doing the current dragging
    this.trail                 = null;      // stores the path to the node that is being dragged
    this.transform             = null;      // transform of the trail to our node (but not including our node, so we can prepend the deltas)
    this.node                  = null;      // the node that we are handling the drag for
    this.lastDragPoint         = null;      // the location of the drag at the previous event (so we can calculate a delta)
    this.startTransformMatrix  = null;      // the node's transform at the start of the drag, so we can reset on a touch cancel
    this.mouseButton           = undefined; // tracks which mouse button was pressed, so we can handle that specifically
    // TODO: consider mouse buttons as separate fingers?
    
    // this listener gets added to the finger when it starts dragging our node
    this.dragListener = {
      // mouse/touch up
      up: function( finger, trail, event ) {
        phet.assert( finger === handler.finger );
        if ( !finger.isMouse || event.button === handler.mouseButton ) {
          handler.endDrag( event );
        }
      },
      
      // touch cancel
      cancel: function( finger, trail, event ) {
        phet.assert( finger === handler.finger );
        handler.endDrag( event );
        
        // since it's a cancel event, go back!
        handler.node.setMatrix( handler.startTransformMatrix );
      },
      
      // mouse/touch move
      move: function( finger, trail, event ) {
        phet.assert( finger === handler.finger );
        // move by the delta between the previous point, using the precomputed transform
        // prepend the translation on the node, so we can ignore whatever other transform state the node has
        handler.node.translate( handler.transform.inverseDelta2( finger.point.minus( handler.lastDragPoint ) ), true );
        handler.lastDragPoint = finger.point;
      }
    };
  };
  var SimpleDragHandler = scenery.SimpleDragHandler;
  
  SimpleDragHandler.prototype = {
    constructor: SimpleDragHandler,
    
    startDrag: function( finger, trail, event, currentTarget ) {
      // set a flag on the finger so it won't pick up other nodes
      finger.dragging = true;
      finger.addInputListener( this.dragListener );
      
      // set all of our persistent information
      this.dragging = true;
      this.finger = finger;
      this.trail = trail.subtrailTo( currentTarget, true );
      this.transform = this.trail.getTransform();
      this.node = currentTarget;
      this.lastDragPoint = finger.point;
      this.startTransformMatrix = currentTarget.getMatrix();
      this.mouseButton = event.button; // should be undefined for touch events
      
      if ( this.options.start ) {
        this.options.start( finger, this.trail, event );
      }
    },
    
    endDrag: function( event ) {
      this.finger.dragging = false;
      this.finger.removeInputListener( this.dragListener );
      this.dragging = false;
      
      if ( this.options.end ) {
        this.options.end( this.finger, this.trail, event );
      }
    },
    
    tryToSnag: function( finger, trail, event, currentTarget ) {
      // only start dragging if the finger isn't dragging anything, we aren't being dragged, and if it's a mouse it's button is down
      if ( !this.dragging && !finger.dragging ) {
        this.startDrag( finger, trail, event, currentTarget );
      }
    },
    
    /*---------------------------------------------------------------------------*
    * events called from the input listener
    *----------------------------------------------------------------------------*/
    
    // mouse/touch down on this node
    down: function( finger, trail, event, currentTarget ) {
      this.tryToSnag( finger, trail, event, currentTarget );
    },
    
    // mouse/touch enters this node
    enter: function( finger, trail, event, currentTarget ) {
      // allow touches to start a drag by moving "over" this node
      if ( this.options.allowTouchSnag && !finger.isMouse ) {
        this.tryToSnag( finger, trail, event, currentTarget );
      }
    }
  };
  
})();


