// Copyright 2002-2013, University of Colorado Boulder

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

  var Bounds2 = require( 'DOT/Bounds2' );
  var Transform3 = require( 'DOT/Transform3' );
  var Matrix3 = require( 'DOT/Matrix3' );

  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/util/Trail' );

  var Util = require( 'SCENERY/util/Util' );

  scenery.PointerOverlay = function PointerOverlay( scene ) {
    var pointerOverlay = this;
    this.scene = scene;

    // add element to show the pointers
    this.pointerSVGContainer = document.createElement( 'div' );
    this.pointerSVGContainer.style.position = 'absolute';
    this.pointerSVGContainer.style.top = 0;
    this.pointerSVGContainer.style.left = 0;
    this.pointerSVGContainer.style['pointer-events'] = 'none';
    this.pointerSVGContainer.style.zIndex = 100;//Make sure it is in front of enough other things!

    var innerRadius = 30;
    var strokeWidth = 10;
    var diameter = (innerRadius + strokeWidth / 2) * 2;
    var radius = diameter / 2;

    //Resize the parent div when the scene is resized
    scene.addEventListener( 'resize', function( args ) {
      pointerOverlay.pointerSVGContainer.setAttribute( 'width', args.width );
      pointerOverlay.pointerSVGContainer.setAttribute( 'height', args.height );
      pointerOverlay.pointerSVGContainer.style.clip = 'rect(0px,' + args.width + 'px,' + args.height + 'px,0px)';
    }, false );

    scene.input.pointerListener = {

      //Display a pointer that was added.  Use a separate SVG layer for each pointer so it can be hardware accelerated, otherwise it is too slow just setting svg internal attributes
      pointerAdded: function( pointer ) {

        var svg = document.createElementNS( 'http://www.w3.org/2000/svg', 'svg' );
        svg.style.position = 'absolute';
        svg.style.top = 0;
        svg.style.left = 0;
        svg.style['pointer-events'] = 'none';

        //Fit the size to the display
        svg.setAttribute( 'width', diameter );
        svg.setAttribute( 'height', diameter );
        var circle = document.createElementNS( 'http://www.w3.org/2000/svg', 'circle' );

        //use css transform for performance?
        circle.setAttribute( 'cx', innerRadius + strokeWidth / 2 );
        circle.setAttribute( 'cy', innerRadius + strokeWidth / 2 );
        circle.setAttribute( 'r', innerRadius );
        circle.setAttribute( 'style', 'stroke:cyan; stroke-width:10; fill:none;' );
        pointer.svg = svg;

        // If there is no point, show the pointer way off screen so that it isn't visible to the user.
        //TODO: remove the need for this workaround
        if ( pointer.point === null ) {

          pointer.point = { x: -10000, y: -1000 };
        }

        //Add a move listener to the pointer to update position when it has moved
        var moveListener = {
          move: function() {

            //TODO: this allocates memory when pointers are dragging, perhaps rewrite to remove allocations
            Util.applyCSSTransform( Matrix3.translation( pointer.point.x - radius, pointer.point.y - radius ), pointer.svg );
          }
        };
        pointer.addInputListener( moveListener );
        pointer.moveListener = moveListener;

        moveListener.move();
        svg.appendChild( circle );
        pointerOverlay.pointerSVGContainer.appendChild( svg );
      },

      pointerRemoved: function( pointer ) {
        pointerOverlay.pointerSVGContainer.removeChild( pointer.svg );
        pointer.removeInputListener( pointer.moveListener );
        delete pointer.moveListener;
        delete pointer.path;
      }
    };

    //if there is a mouse, add it here
    if ( scene.input && scene.input.mouse ) {
      scene.input.pointerListener.pointerAdded( scene.input.mouse );
    }

    scene.$main[0].appendChild( this.pointerSVGContainer );
  };
  var PointerOverlay = scenery.PointerOverlay;

  PointerOverlay.prototype = {
    dispose: function() {
      this.scene.$main[0].removeChild( this.pointerSVGContainer );
      delete this.scene.input.pointerListener;
    }
  };

  return PointerOverlay;
} );