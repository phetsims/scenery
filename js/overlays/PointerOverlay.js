// Copyright 2002-2014, University of Colorado Boulder

/**
 * The PointerOverlay shows pointer locations in the scene.  This is useful when recording a session for interviews or when a teacher is broadcasting
 * a tablet session on an overhead projector.  See https://github.com/phetsims/scenery/issues/111
 *
 * Each pointer is rendered in a different <svg> so that CSS3 transforms can be used to make performance smooth on iPad.
 *
 * @author Sam Reid
 */

define( function( require ) {
  'use strict';

  var Matrix3 = require( 'DOT/Matrix3' );

  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/util/Trail' );

  var Util = require( 'SCENERY/util/Util' );

  scenery.PointerOverlay = function PointerOverlay( display, scene ) {
    var pointerOverlay = this;
    this.display = display;
    this.scene = scene;

    // add element to show the pointers
    this.pointerSVGContainer = document.createElement( 'div' );
    this.pointerSVGContainer.style.position = 'absolute';
    this.pointerSVGContainer.style.top = 0;
    this.pointerSVGContainer.style.left = 0;
    this.pointerSVGContainer.style['pointer-events'] = 'none';

    var innerRadius = 10;
    var strokeWidth = 1;
    var diameter = (innerRadius + strokeWidth / 2) * 2;
    var radius = diameter / 2;

    //Resize the parent div when the scene is resized
    display.onStatic( 'displaySize', function( dimension ) {
      pointerOverlay.pointerSVGContainer.setAttribute( 'width', dimension.width );
      pointerOverlay.pointerSVGContainer.setAttribute( 'height', dimension.height );
      pointerOverlay.pointerSVGContainer.style.clip = 'rect(0px,' + dimension.width + 'px,' + dimension.height + 'px,0px)';
    } );

    //Display a pointer that was added.  Use a separate SVG layer for each pointer so it can be hardware accelerated, otherwise it is too slow just setting svg internal attributes
    var pointerAdded = this.pointerAdded = function( pointer ) {

      if ( pointer.isKey ) { return; }

      var svg = document.createElementNS( scenery.svgns, 'svg' );
      svg.style.position = 'absolute';
      svg.style.top = 0;
      svg.style.left = 0;
      svg.style['pointer-events'] = 'none';

      //Fit the size to the display
      svg.setAttribute( 'width', diameter );
      svg.setAttribute( 'height', diameter );

      var circle = document.createElementNS( scenery.svgns, 'circle' );

      //use css transform for performance?
      circle.setAttribute( 'cx', innerRadius + strokeWidth / 2 );
      circle.setAttribute( 'cy', innerRadius + strokeWidth / 2 );
      circle.setAttribute( 'r', innerRadius );
      circle.setAttribute( 'style', 'fill:black;' );
      circle.setAttribute( 'opacity', 0.4 );

      //Add a move listener to the pointer to update position when it has moved
      var pointerRemoved = function() {

        //For touches that get a touch up event, remove them.  But when the mouse button is released, don't stop showing the mouse location
        if ( pointer.isTouch ) {
          pointerOverlay.pointerSVGContainer.removeChild( svg );
          pointer.removeInputListener( moveListener );
        }
      };
      var moveListener = {
        move: function() {

          //TODO: Why is point sometimes null?
          if ( pointer.point ) {

            //TODO: this allocates memory when pointers are dragging, perhaps rewrite to remove allocations
            Util.applyCSSTransform( Matrix3.translation( pointer.point.x - radius, pointer.point.y - radius ), svg );
          }
        },

        up: pointerRemoved,
        cancel: pointerRemoved
      };
      pointer.addInputListener( moveListener );

      moveListener.move();
      svg.appendChild( circle );
      pointerOverlay.pointerSVGContainer.appendChild( svg );
    };
    display._input.addPointerAddedListener( pointerAdded );

    //if there is already a mouse, add it here
    //TODO: if there already other non-mouse touches, could be added here
    if ( display._input && display._input.mouse ) {
      pointerAdded( display._input.mouse );
    }
    
    this.domElement = this.pointerSVGContainer;
  };
  var PointerOverlay = scenery.PointerOverlay;

  PointerOverlay.prototype = {
    dispose: function() {
      this.display._input.removePointerAddedListener( this.pointerAdded );
    },
    
    update: function() {
      
    }
  };

  return PointerOverlay;
} );
