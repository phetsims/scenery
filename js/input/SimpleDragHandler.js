// Copyright 2002-2013, University of Colorado

/**
 * Basic dragging for a node.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var scenery = require( 'SCENERY/scenery' );
  
  /*
   * Allowed options: {
   *    allowTouchSnag: false // allow touch swipes across an object to pick it up,
   *    dragCursor: 'pointer' // while dragging with the mouse, sets the cursor to this value (or use null to not override the cursor while dragging)
   *    mouseButton: 0        // allow changing the mouse button that activates the drag listener. -1 should activate on any mouse button, 0 on left, 1 for middle, 2 for right, etc.
   *    start: null           // if non-null, called when a drag is started. start( event, trail )
   *    drag: null            // if non-null, called when the user moves something with a drag (not a start or end event).
   *                                                                         drag( event, trail )
   *    end: null             // if non-null, called when a drag is ended.   end( event, trail )
   *    translate:            // if this exists, translate( { delta: _, oldPosition: _, position: _ } ) will be called.
   * }
   */
  scenery.SimpleDragHandler = function SimpleDragHandler( options ) {
    var handler = this;
    
    this.options = _.extend( {
      allowTouchSnag: false,
      mouseButton: 0,
      dragCursor: 'pointer'
    }, options );
    
    this.dragging              = false;     // whether a node is being dragged with this handler
    this.pointer               = null;      // the pointer doing the current dragging
    this.trail                 = null;      // stores the path to the node that is being dragged
    this.transform             = null;      // transform of the trail to our node (but not including our node, so we can prepend the deltas)
    this.node                  = null;      // the node that we are handling the drag for
    this.lastDragPoint         = null;      // the location of the drag at the previous event (so we can calculate a delta)
    this.startTransformMatrix  = null;      // the node's transform at the start of the drag, so we can reset on a touch cancel
    this.mouseButton           = undefined; // tracks which mouse button was pressed, so we can handle that specifically
    // TODO: consider mouse buttons as separate pointers?
    
    // if an ancestor is transformed, pin our node
    this.transformListener = {
      transform: function( args ) {
        if ( !handler.trail.isExtensionOf( args.trail, true ) ) {
          return;
        }
        
        var newMatrix = args.trail.getTransform().getMatrix();
        var oldMatrix = handler.transform.getMatrix();
        
        // if A was the trail's old transform, B is the trail's new transform, we need to apply (B^-1 A) to our node
        handler.node.prependMatrix( newMatrix.inverted().timesMatrix( oldMatrix ) );
        
        // store the new matrix so we can do deltas using it now
        handler.transform.setMatrix( newMatrix );
      }
    };
    
    // this listener gets added to the pointer when it starts dragging our node
    this.dragListener = {
      // mouse/touch up
      up: function( event ) {
        assert && assert( event.pointer === handler.pointer );
        if ( !event.pointer.isMouse || event.domEvent.button === handler.mouseButton ) {
          var saveCurrentTarget = event.currentTarget;
          event.currentTarget = handler.node; // #66: currentTarget on a pointer is null, so set it to the node we're dragging
          handler.endDrag( event );
          event.currentTarget = saveCurrentTarget; // be polite to other listeners, restore currentTarget
        }
      },
      
      // touch cancel
      cancel: function( event ) {
        assert && assert( event.pointer === handler.pointer );

        var saveCurrentTarget = event.currentTarget;
        event.currentTarget = handler.node; // #66: currentTarget on a pointer is null, so set it to the node we're dragging
        handler.endDrag( event );
        event.currentTarget = saveCurrentTarget; // be polite to other listeners, restore currentTarget

        // since it's a cancel event, go back!
        if ( !handler.transform ) {
          handler.node.setMatrix( handler.startTransformMatrix );
        }
      },
      
      // mouse/touch move
      move: function( event ) {
        assert && assert( event.pointer === handler.pointer );
        
        var delta = handler.transform.inverseDelta2( handler.pointer.point.minus( handler.lastDragPoint ) );
        
        // move by the delta between the previous point, using the precomputed transform
        // prepend the translation on the node, so we can ignore whatever other transform state the node has
        if ( handler.options.translate ) {
          var translation = handler.node.getTransform().getMatrix().getTranslation();
          handler.options.translate( {
            delta: delta,
            oldPosition: translation,
            position: translation.plus( delta )
          } );
        }
        handler.lastDragPoint = handler.pointer.point;
        
        if ( handler.options.drag ) {
          // TODO: consider adding in a delta to the listener
          // TODO: add the position in to the listener
          var saveCurrentTarget = event.currentTarget;
          event.currentTarget = handler.node; // #66: currentTarget on a pointer is null, so set it to the node we're dragging
          handler.options.drag( event, handler.trail ); // new position (old position?) delta
          event.currentTarget = saveCurrentTarget; // be polite to other listeners, restore currentTarget
        }
      }
    };
  };
  var SimpleDragHandler = scenery.SimpleDragHandler;
  
  SimpleDragHandler.prototype = {
    constructor: SimpleDragHandler,
    
    startDrag: function( event ) {
      // set a flag on the pointer so it won't pick up other nodes
      event.pointer.dragging = true;
      event.pointer.cursor = this.options.dragCursor;
      event.pointer.addInputListener( this.dragListener );
      // event.trail.rootNode().addEventListener( this.transformListener ); // TODO: replace with new parent transform listening solution
      
      // set all of our persistent information
      this.dragging = true;
      this.pointer = event.pointer;
      this.trail = event.trail.subtrailTo( event.currentTarget, true );
      this.transform = this.trail.getTransform();
      this.node = event.currentTarget;
      this.lastDragPoint = event.pointer.point;
      this.startTransformMatrix = event.currentTarget.getMatrix();
      // event.domEvent may not exist if this is touch-to-snag
      this.mouseButton = event.pointer.isMouse ? event.domEvent.button : undefined;
      
      if ( this.options.start ) {
        this.options.start( event, this.trail );
      }
    },
    
    endDrag: function( event ) {
      this.pointer.dragging = false;
      this.pointer.cursor = null;
      this.pointer.removeInputListener( this.dragListener );
      // this.trail.rootNode().removeEventListener( this.transformListener ); // TODO: replace with new parent transform listening solution
      this.dragging = false;
      
      if ( this.options.end ) {
        this.options.end( event, this.trail );
      }
    },
    
    tryToSnag: function( event ) {
      // don't allow drag attempts that use the wrong mouse button (-1 indicates any mouse button works)
      if ( event.pointer.isMouse && event.domEvent && this.options.mouseButton !== event.domEvent.button && this.options.mouseButton !== -1 ) {
        return;
      }
      
      // only start dragging if the pointer isn't dragging anything, we aren't being dragged, and if it's a mouse it's button is down
      if ( !this.dragging && !event.pointer.dragging ) {
        this.startDrag( event );
      }
    },
    
    /*---------------------------------------------------------------------------*
    * events called from the node input listener
    *----------------------------------------------------------------------------*/
    
    // mouse/touch down on this node
    down: function( event ) {
      this.tryToSnag( event );
    },
    
    // touch enters this node
    touchenter: function( event ) {
      // allow touches to start a drag by moving "over" this node
      if ( this.options.allowTouchSnag ) {
        this.tryToSnag( event );
      }
    }
  };
  
  return SimpleDragHandler;
} );


