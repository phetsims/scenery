// Copyright 2002-2012, University of Colorado

/**
 * Main scene, that is also a Node.
 *
 * TODO: documentation!
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var Bounds2 = require( 'DOT/Bounds2' );
  var Vector2 = require( 'DOT/Vector2' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Node = require( 'SCENERY/Node' ); // inherits from Node
  require( 'SCENERY/Trail' );
  require( 'SCENERY/TrailPointer' );
  require( 'SCENERY/input/Input' );
  require( 'SCENERY/layers/LayerBuilder' );
  require( 'SCENERY/layers/Renderer' );
  
  var Util = require( 'SCENERY/util/Util' );
  var objectCreate = Util.objectCreate;
  
  /*
   * $main should be a block-level element with a defined width and height. scene.resize() should be called whenever
   * it is resized.
   *
   * Valid parameters in the parameter object:
   * {
   *   allowSceneOverflow: false,           // usually anything displayed outside of this $main (DOM/CSS3 transformed SVG) is hidden with CSS overflow
   *   allowCSSHacks: true,                 // applies styling that prevents mobile browser graphical issues
   *   allowDevicePixelRatioScaling: false, // allows underlying canvases (Canvas, WebGL) to increase in size to maintain sharpness on high-density displays
   *   preferredSceneLayerType: ...,        // sets the preferred type of layer to be created if there are multiple options
   *   width: <current main width>,         // override the main container's width
   *   height: <current main height>,       // override the main container's height
   * }
   */
  scenery.Scene = function( $main, options ) {
    assert && assert( $main[0], 'A main container is required for a scene' );
    
    // defaults
    options = _.extend( {
      allowSceneOverflow: false,
      allowCSSHacks: true,
      allowDevicePixelRatioScaling: false,
      preferredSceneLayerType: scenery.CanvasDefaultLayerType,
      width: $main.width(),
      height: $main.height()
    }, options || {} );
    
    this.backingScale = options.allowDevicePixelRatioScaling ? Util.backingScale( document.createElement( 'canvas' ).getContext( '2d' ) ) : 1;
    
    Node.call( this, options );
    
    var scene = this;
    
    // main layers in a scene
    this.layers = [];
    
    this.lastCursor = null;
    this.defaultCursor = $main.css( 'cursor' );
    
    this.$main = $main;
    // resize the main container as a sanity check
    this.$main.width( options.width );
    this.$main.height( options.height );
    
    this.sceneBounds = new Bounds2( 0, 0, $main.width(), $main.height() );
    
    // default to a canvas layer type, but this can be changed
    this.preferredSceneLayerType = options.preferredSceneLayerType;
    
    applyCSSHacks( $main, options );
    
    // note, arguments to the functions are mutable. don't destroy them
    this.sceneEventListener = {
      insertChild: function( args ) {
        // var parent = args.parent;
        // var child = args.child;
        // var index = args.index;
        // var trail = args.trail;
        
        // TODO: improve later
        scene.rebuildLayers();
      },
      
      removeChild: function( args ) {
        // var parent = args.parent;
        // var child = args.child;
        // var index = args.index;
        // var trail = args.trail;
        
        scene.rebuildLayers();
      },
      
      dirtyBounds: function( args ) {
        // var node = args.node;
        // var localBounds = args.bounds;
        // var transform = args.transform;
        var trail = args.trail;
        
        // if there are no layers, no nodes would actually render, so don't do the lookup
        if ( scene.layers.length ) {
          _.each( scene.affectedLayers( trail ), function( layer ) {
            layer.markDirtyRegion( args );
          } );
        }
      },
      
      transform: function( args ) {
        // var node = args.node;
        // var type = args.type;
        // var matrix = args.matrix;
        // var transform = args.transform;
        var trail = args.trail;
        
        if ( scene.layers.length ) {
          _.each( scene.affectedLayers( trail ), function( layer ) {
            layer.transformChange( args );
          } );
        }
      },
      
      layerRefresh: function( args ) {
        // var node = args.node;
        // var trail = args.trail;
        
        scene.rebuildLayers();
      }
    };
    
    this.addEventListener( this.sceneEventListener );
  };
  var Scene = scenery.Scene;

  Scene.prototype = objectCreate( Node.prototype );
  Scene.prototype.constructor = Scene;
  
  Scene.prototype.updateScene = function( args ) {
    // validating bounds, similar to Piccolo2d
    this.validateBounds();
    this.validatePaint();
    
    // bail if there are no layers. consider a warning?
    if ( !this.layers.length ) {
      return;
    }
    
    var scene = this;
    
    _.each( this.layers, function( layer ) {
      layer.render( scene, args );
    } );
    
    this.updateCursor();
  };
  
  Scene.prototype.renderScene = function() {
    // TODO: for now, go with the same path. possibly add options later
    this.updateScene();
  };
  
  Scene.prototype.rebuildLayers = function() {
    // remove all of our tracked layers from the container, so we can fill it with fresh layers
    this.disposeLayers();
    
    // TODO: internal API rethink
    var state = new scenery.LayerBuilder();
    
    if ( this.preferredSceneLayerType ) {
      state.pushPreferredLayerType( this.preferredSceneLayerType );
    }
    
    var layerEntries = state.buildLayers( new scenery.TrailPointer( new scenery.Trail( this ), true ), new scenery.TrailPointer( new scenery.Trail( this ), false ), null );
    
    var layerArgs = {
      $main: this.$main,
      scene: this,
      baseNode: this
    };
    
    this.layers = _.map( layerEntries, function( entry ) {
      var layer = entry.type.createLayer( _.extend( {}, layerArgs, entry.type.args ), entry );
      return layer;
    } );
    
    // console.log( '---' );
    // console.log( 'layers rebuilt:' );
    // _.each( this.layers, function( layer ) {
    //   console.log( layer.toString() );
    // } );
    // _.each( layerEntries, function( entry ) {
    //   //console.log( entry.type.name + ' ' + ( this.startPointer ? this.startPointer.toString() : '!' ) + ' (' + ( this.startPath ? this.startPath.toString() : '!' ) + ') => ' + ( this.endPointer ? this.endPointer.toString() : '!' ) + ' (' + ( this.endPath ? this.endPath.toString() : '!' ) + ')' );
    // } );
  };
  
  // after layer changes, the layers should have their zIndex updated
  Scene.prototype.reindexLayers = function() {
    var index = 1;
    _.each( this.layers, function( layer ) {
      // layers increment indices as needed
      index = layer.reindex( index );
    } );
  };
  
  Scene.prototype.dispose = function() {
    this.disposeLayers();
    
    // TODO: clear event handlers if added
    //throw new Error( 'unimplemented dispose: clear event handlers if added' );
  };
  
  Scene.prototype.disposeLayers = function() {
    var scene = this;
    
    _.each( this.layers.slice( 0 ), function( layer ) {
      layer.dispose();
      scene.layers.splice( _.indexOf( scene.layers, layer ), 1 ); // TODO: better removal code!
    } );
  };
  
  // what layer does this trail's terminal node render in?
  Scene.prototype.layerLookup = function( trail ) {
    // TODO: add tree form for optimization! this is slower than necessary, it shouldn't be O(n)!
    
    assert && assert( !( trail.isEmpty() || trail.nodes[0] !== this ), 'layerLookup root matches' );
    
    if ( this.layers.length === 0 ) {
      throw new Error( 'no layers in the scene' );
    }
    
    // point to the beginning of the node, right before it would be rendered
    var pointer = new scenery.TrailPointer( trail, true );
    
    for ( var i = 0; i < this.layers.length; i++ ) {
      var layer = this.layers[i];
      
      // the first layer whose end point is equal to or past our pointer should contain the trail
      if ( pointer.compareNested( layer.endPointer ) !== 1 ) {
        // TODO: consider removal for performance
        assert && assert( pointer.compareNested( layer.startPointer ) !== -1, 'node not contained in a layer' );
        return layer;
      }
    }
    
    throw new Error( 'node not contained in a layer' );
  };
  
  // all layers whose start or end points lie inclusively in the range from the trail's before and after
  Scene.prototype.affectedLayers = function( trail ) {
    // TODO: add tree form for optimization! this is slower than necessary, it shouldn't be O(n)!
    
    var result = [];
    
    assert && assert( !( trail.isEmpty() || trail.nodes[0] !== this ), 'layerLookup root matches' );
    
    if ( this.layers.length === 0 ) {
      throw new Error( 'no layers in the scene' );
    }
    
    // point to the beginning of the node, right before it would be rendered
    var startPointer = new scenery.TrailPointer( trail, true );
    var endPointer = new scenery.TrailPointer( trail, false );
    
    for ( var i = 0; i < this.layers.length; i++ ) {
      var layer = this.layers[i];
      
      var notBefore = endPointer.compareNested( layer.startPointer ) !== -1;
      var notAfter = startPointer.compareNested( layer.endPointer ) !== 1;
      
      if ( notBefore && notAfter ) {
        result.push( layer );
      }
    }
    
    return result;
  };
  
  // attempt to render everything currently visible in the scene to an external canvas. allows copying from canvas layers straight to the other canvas
  Scene.prototype.renderToCanvas = function( canvas, context, callback ) {
    var count = 0;
    var started = false; // flag guards against asynchronous tests that call back synchronously (immediate increment and decrement)
    var delayCounts = {
      increment: function() {
        count++;
      },
      
      decrement: function() {
        count--;
        if ( count === 0 && callback && started ) {
          callback();
        }
      }
    };
    
    context.clearRect( 0, 0, canvas.width, canvas.height );
    _.each( this.layers, function( layer ) {
      layer.renderToCanvas( canvas, context, delayCounts );
    } );
    
    if ( count === 0 ) {
      // no asynchronous layers, callback immediately
      if ( callback ) {
        callback();
      }
    } else {
      started = true;
    }
  };
  
  // TODO: consider SVG data URLs
  
  Scene.prototype.canvasDataURL = function( callback ) {
    this.canvasSnapshot( function( canvas ) {
      callback( canvas.toDataURL() );
    } );
  };
  
  // renders what it can into a Canvas (so far, Canvas and SVG layers work fine)
  Scene.prototype.canvasSnapshot = function( callback ) {
    var canvas = document.createElement( 'canvas' );
    canvas.width = this.sceneBounds.width();
    canvas.height = this.sceneBounds.height();
    
    var context = canvas.getContext( '2d' );
    this.renderToCanvas( canvas, context, function() {
      callback( canvas, context.getImageData( 0, 0, canvas.width, canvas.height ) );
    } );
  };
  
  Scene.prototype.resize = function( width, height ) {
    this.$main.width( width );
    this.$main.height( height );
    this.sceneBounds = new Bounds2( 0, 0, width, height );
    this.rebuildLayers(); // TODO: why?
  };
  
  Scene.prototype.updateCursor = function() {
    if ( this.input && this.input.mouse.point ) {
      var mouseTrail = this.trailUnderPoint( this.input.mouse.point );
      
      if ( mouseTrail ) {
        for ( var i = mouseTrail.length - 1; i >= 0; i-- ) {
          var cursor = mouseTrail.nodes[i].getCursor();
          
          if ( cursor ) {
            this.setSceneCursor( cursor );
            return;
          }
        }
      }
    }
    
    // fallback case
    this.setSceneCursor( this.defaultCursor );
  };
  
  Scene.prototype.setSceneCursor = function( cursor ) {
    if ( cursor !== this.lastCursor ) {
      this.lastCursor = cursor;
      this.$main.css( 'cursor', cursor );
    }
  };
  
  Scene.prototype.initializeStandaloneEvents = function( parameters ) {
    var element = this.$main[0];
    this.initializeEvents( _.extend( {}, {
      preventDefault: true,
      listenerTarget: element,
      pointFromEvent: function( evt ) {
        var mainBounds = element.getBoundingClientRect();
        return new Vector2( evt.clientX - mainBounds.left, evt.clientY - mainBounds.top );
      }
    }, parameters ) );
  };
  
  Scene.prototype.initializeFullscreenEvents = function( parameters ) {
    this.initializeEvents( _.extend( {}, {
      preventDefault: true,
      listenerTarget: document,
      pointFromEvent: function( evt ) {
        return new Vector2( evt.pageX, evt.pageY );
      }
    }, parameters ) );
  };
  
  Scene.prototype.initializeEvents = function( parameters ) {
    var scene = this;
    
    if ( scene.input ) {
      throw new Error( 'Attempt to attach events twice to the scene' );
    }
    
    // TODO: come up with more parameter names that have the same string length, so it looks creepier
    var pointFromEvent = parameters.pointFromEvent;
    var listenerTarget = parameters.listenerTarget;
    var preventDefault = parameters.preventDefault;
    
    var input = new scenery.Input( scene );
    scene.input = input;
    
    $( listenerTarget ).on( 'mousedown', function( jEvent ) {
      var evt = jEvent.originalEvent;
      if ( preventDefault ) { jEvent.preventDefault(); }
      input.mouseDown( pointFromEvent( evt ), evt );
    } );
    $( listenerTarget ).on( 'mouseup', function( jEvent ) {
      var evt = jEvent.originalEvent;
      if ( preventDefault ) { jEvent.preventDefault(); }
      input.mouseUp( pointFromEvent( evt ), evt );
    } );
    $( listenerTarget ).on( 'mousemove', function( jEvent ) {
      var evt = jEvent.originalEvent;
      if ( preventDefault ) { jEvent.preventDefault(); }
      input.mouseMove( pointFromEvent( evt ), evt );
    } );
    $( listenerTarget ).on( 'mouseover', function( jEvent ) {
      var evt = jEvent.originalEvent;
      if ( preventDefault ) { jEvent.preventDefault(); }
      input.mouseOver( pointFromEvent( evt ), evt );
    } );
    $( listenerTarget ).on( 'mouseout', function( jEvent ) {
      var evt = jEvent.originalEvent;
      if ( preventDefault ) { jEvent.preventDefault(); }
      input.mouseOut( pointFromEvent( evt ), evt );
    } );

    function forEachChangedTouch( evt, callback ) {
      for ( var i = 0; i < evt.changedTouches.length; i++ ) {
        // according to spec (http://www.w3.org/TR/touch-events/), this is not an Array, but a TouchList
        var touch = evt.changedTouches.item( i );
        
        callback( touch.identifier, pointFromEvent( touch ) );
      }
    }

    $( listenerTarget ).on( 'touchstart', function( jEvent ) {
      var evt = jEvent.originalEvent;
      if ( preventDefault ) { jEvent.preventDefault(); }
      forEachChangedTouch( evt, function( id, point ) {
        input.touchStart( id, point, evt );
      } );
    } );
    $( listenerTarget ).on( 'touchend', function( jEvent ) {
      var evt = jEvent.originalEvent;
      if ( preventDefault ) { jEvent.preventDefault(); }
      forEachChangedTouch( evt, function( id, point ) {
        input.touchEnd( id, point, evt );
      } );
    } );
    $( listenerTarget ).on( 'touchmove', function( jEvent ) {
      var evt = jEvent.originalEvent;
      if ( preventDefault ) { jEvent.preventDefault(); }
      forEachChangedTouch( evt, function( id, point ) {
        input.touchMove( id, point, evt );
      } );
    } );
    $( listenerTarget ).on( 'touchcancel', function( jEvent ) {
      var evt = jEvent.originalEvent;
      if ( preventDefault ) { jEvent.preventDefault(); }
      forEachChangedTouch( evt, function( id, point ) {
        input.touchCancel( id, point, evt );
      } );
    } );
  };
  
  Scene.prototype.resizeOnWindowResize = function() {
    var scene = this;
    
    var resizer = function () {
      scene.resize( window.innerWidth, window.innerHeight );
    };
    $( window ).resize( resizer );
    resizer();
  };
    
  function applyCSSHacks( $main, options ) {
    // to use CSS3 transforms for performance, hide anything outside our bounds by default
    if ( options.allowSceneOverflow ) {
      $main.css( 'overflow', 'hidden' );
    }
    
    if ( options.allowCSSHacks ) {
      // some css hacks (inspired from https://github.com/EightMedia/hammer.js/blob/master/hammer.js)
      (function() {
        var prefixes = [ '-webkit-', '-moz-', '-ms-', '-o-', '' ];
        var properties = {
          userSelect: 'none',
          touchCallout: 'none',
          touchAction: 'none',
          userDrag: 'none',
          tapHighlightColor: 'rgba(0,0,0,0)'
        };
        
        _.each( prefixes, function( prefix ) {
          _.each( properties, function( propertyValue, propertyName ) {
            $main.css( prefix + propertyName, propertyValue );
          } );
        } );
      })();
    }
  }
  
  return Scene;
} );
