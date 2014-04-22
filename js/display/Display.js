// Copyright 2002-2013, University of Colorado

/**
 * A persistent display of a specific Node and its descendants, which is updated at discrete points in time.
 *
 * Use display.getDOMElement or display.domElement to retrieve the Display's DOM representation.
 * Use display.updateDisplay() to trigger the visual update in the Display's DOM element.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Dimension2 = require( 'DOT/Dimension2' );
  require( 'SCENERY/util/Trail' );
  require( 'SCENERY/display/BackboneBlock' );
  require( 'SCENERY/display/CanvasBlock' );
  require( 'SCENERY/display/CanvasSelfDrawable' );
  require( 'SCENERY/display/DisplayInstance' );
  require( 'SCENERY/display/DOMSelfDrawable' );
  require( 'SCENERY/display/InlineCanvasCacheDrawable' );
  require( 'SCENERY/display/RenderState' );
  require( 'SCENERY/display/SharedCanvasCacheDrawable' );
  require( 'SCENERY/display/SVGSelfDrawable' );
  require( 'SCENERY/layers/Renderer' );
  
  // Constructs a Display that will show the rootNode and its subtree in a visual state. Default options provided below
  scenery.Display = function Display( rootNode, options ) {
    this.options = _.extend( {
      width: 640,               // initial display width
      height: 480,              // initial display height
      //OHTWO TODO: hook up allowCSSHacks
      allowCSSHacks: true,      // applies CSS styles to the root DOM element that make it amenable to interactive content
      //OHTWO TODO: hook up enablePointerEvents
      enablePointerEvents: true // whether we should specifically listen to pointer events if we detect support
    }, options );
    
    // The (integral, > 0) dimensions of the Display's DOM element (only updates the DOM element on updateDisplay())
    this._size = new Dimension2( this.options.width, this.options.height );
    
    this._rootNode = rootNode;
    this._rootBackbone = null; // to be filled in later
    this._domElement = scenery.BackboneBlock.createDivBackbone();
    this._sharedCanvasInstances = {}; // map from Node ID to DisplayInstance, for fast lookup
    this._baseInstance = null; // will be filled with the root DisplayInstance
    
    // variable state
    this._frameId = 0; // incremented for every rendered frame
    this._dirtyTransformRoots = [];
    this._dirtyTransformRootsWithoutPass = [];
  };
  var Display = scenery.Display;
  
  // display instance linked list ops
  function connectDrawables( a, b ) {
    a.nextDrawable = b;
    b.previousDrawable = a;
  }
  
  function createInstance( display, trail, state, parentInstance ) {
    var instance = new scenery.DisplayInstance( display, trail, state );
    
    var node = trail.lastNode();
    // instance.parent = parentInstance; // NOTE: done later during the append!
    instance.isTransformed = state.isTransformed;
    if ( instance.isTransformed ) {
      display.markTransformRootDirty( instance, true );
    }
    
    if ( state.isCanvasCache && state.isCacheShared ) {
      var instanceKey = trail.lastNode().getId();
      var sharedInstance = display._sharedCanvasInstances[instanceKey];
      
      // TODO: assert state is the same?
      // TODO: increment reference counting?
      if ( !sharedInstance ) {
        var sharedNode = trail.lastNode();
        sharedInstance = createInstance( display, new scenery.Trail( sharedNode ), scenery.RenderState.RegularState.createSharedCacheState( sharedNode ), null );
        display._sharedCanvasInstances[instanceKey] = sharedInstance;
        // TODO: reference counting?
        
        // TODO: sharedInstance.isTransformed?
      }
      
      // TODO: do something with the sharedInstance!
      var sharedCacheRenderer = state.sharedCacheRenderer;
      // TODO: improve this!
      instance.sharedCacheDrawable = new scenery.SharedCanvasCacheDrawable( trail, sharedCacheRenderer, instance, sharedInstance );
    } else {
      var currentDrawable = null;
      
      if ( node.isPainted() ) {
        var Renderer = scenery.Renderer;
        
        var selfRenderer = state.selfRenderer;
        var selfRendererType = selfRenderer & Renderer.bitmaskRendererArea;
        if ( selfRendererType === Renderer.bitmaskCanvas ) {
          instance.selfDrawable = new scenery.CanvasSelfDrawable( selfRenderer, instance );
        } else if ( selfRendererType === Renderer.bitmaskSVG ) {
          instance.selfDrawable = new scenery.SVGSelfDrawable( selfRenderer, instance );
        } else if ( selfRendererType === Renderer.bitmaskDOM ) {
          instance.selfDrawable = new scenery.DOMSelfDrawable( selfRenderer, instance );
        } else {
          // assert so that it doesn't compile down to a throw (we want this function to be optimized)
          assert && assert( 'Unrecognized renderer, maybe we don\'t support WebGL yet?: ' + selfRenderer );
        }
        instance.firstDrawable = currentDrawable = instance.selfDrawable;
      }
      
      var children = trail.lastNode().children;
      var numChildren = children.length;
      for ( var i = 0; i < numChildren; i++ ) {
        // create a child instance
        var child = children[i];
        var childInstance = createInstance( display, trail.copy().addDescendant( child, i ), state.getStateForDescendant( child ), instance );
        instance.appendInstance( childInstance );
        
        // figure out what the first and last drawable should be hooked into for the child
        var firstChildDrawable = null;
        var lastChildDrawable = null;
        if ( childInstance.groupDrawable ) {
          // if there is a group (e.g. non-shared cache or backbone), use it
          firstChildDrawable = lastChildDrawable = childInstance.groupDrawable;
        } else if ( childInstance.sharedCacheDrawable ) {
          // if there is a shared cache drawable, use it
          firstChildDrawable = lastChildDrawable = childInstance.sharedCacheDrawable;
        } else if ( childInstance.firstDrawable ) {
          // otherwise, if they exist, pick the node's first/last directly
          assert && assert( childInstance.lastDrawable, 'Any display instance with firstDrawable should also have lastDrawable' );
          firstChildDrawable = childInstance.firstDrawable;
          lastChildDrawable = childInstance.lastDrawable;
        }
        
        // if there are any drawables for that child, link them up in our linked list
        if ( firstChildDrawable ) {
          if ( currentDrawable ) {
            // there is already an end of the linked list, so just append to it
            connectDrawables( currentDrawable, firstChildDrawable );
          } else {
            // start out the linked list
            instance.firstDrawable = firstChildDrawable;
          }
          // update the last drawable of the linked list
          currentDrawable = lastChildDrawable;
        }
      }
      if ( currentDrawable !== null ) {
        // finish setting up references to the linked list (now firstDrawable and lastDrawable should be set properly)
        instance.lastDrawable = currentDrawable;
      }
      
      var groupRenderer = state.groupRenderer;
      if ( state.isBackbone ) {
        assert && assert( !state.isCanvasCache, 'For now, disallow an instance being a backbone and a canvas cache, since it has no performance benefits' );
        
        var backbone = new scenery.BackboneBlock( instance, groupRenderer, false, null );
        instance.block = backbone;
        instance.groupDrawable = backbone.blockDrawable;
      } else if ( state.isCanvasCache ) {
        
        // TODO: create block for cache
        // TODO create non-shared cache, use groupRenderer
        instance.groupDrawable = new scenery.InlineCanvasCacheDrawable( trail, groupRenderer, instance );
      }
    }
    
    return instance;
  }
  
  inherit( Object, Display, {
    // returns the base DOM element that will be displayed by this Display
    getDOMElement: function() {
      return this._domElement;
    },
    get domElement() { return this.getDOMElement(); },
    
    // updates the display's DOM element with the current visual state of the attached root node and its descendants
    updateDisplay: function() {
      throw new Error( 'TODO OHTWO unimplemented' );
    },
    
    getRootNode: function() {
      return this._rootNode;
    },
    get rootNode() { return this.getRootNode(); },
    
    // The dimensions of the Display's DOM element
    getSize: function() {
      return this._size;
    },
    get size() { return this.getSize(); },
    
    // size: dot.Dimension2. Changes the size that the Display's DOM element will be after the next updateDisplay()
    setSize: function( size ) {
      assert && assert( size instanceof Dimension2 );
      assert && assert( size.width % 1 === 0, 'Display.width should be an integer' );
      assert && assert( size.width > 0, 'Display.width should be greater than zero' );
      assert && assert( size.height % 1 === 0, 'Display.height should be an integer' );
      assert && assert( size.height > 0, 'Display.height should be greater than zero' );
      
      if ( !this._size.equals( size ) ) {
        this._size = size;
        
        //TODO OHTWO send event about size change, or mark a dirty flag?
      }
    },
    
    // The width of the Display's DOM element
    getWidth: function() {
      return this._size.width;
    },
    get width() { return this.getWidth(); },
    
    // Sets the width that the Display's DOM element will be after the next updateDisplay(). Should be an integral value.
    setWidth: function( width ) {
      assert && assert( typeof width === 'number', 'Display.width should be a number' );
      
      if ( this.getWidth() !== width ) {
        // TODO: remove allocation here?
        this.setSize( new Dimension2( width, this.getHeight() ) );
      }
    },
    set width( value ) { this.setWidth( value ); },
    
    // The height of the Display's DOM element
    getHeight: function() {
      return this._size.height;
    },
    get height() { return this.getHeight(); },
    
    // Sets the height that the Display's DOM element will be after the next updateDisplay(). Should be an integral value.
    setHeight: function( height ) {
      assert && assert( typeof height === 'number', 'Display.height should be a number' );
      
      if ( this.getHeight() !== height ) {
        // TODO: remove allocation here?
        this.setSize( new Dimension2( height, this.getHeight() ) );
      }
    },
    set height( value ) { this.setHeight( value ); },
    
    // called from DisplayInstances that will need a transform update (for listeners and precomputation)
    markTransformRootDirty: function( displayInstance, passTransform ) {
      passTransform ? this._dirtyTransformRoots.push( displayInstance ) : this._dirtyTransformRootsWithoutPass.push( displayInstance );
    },
    
    updateDirtyTransformRoots: function() {
      var len,i;
      
      len = this._dirtyTransformRoots.length;
      for ( i = 0; i < len; i++ ) {
        this._dirtyTransformRoots[i].updateTransformListenersAndCompute( false, false, this._frameId, true );
      }
      this._dirtyTransformRoots.length = 0;
      
      len = this._dirtyTransformRootsWithoutPass.length;
      for ( i = 0; i < len; i++ ) {
        this._dirtyTransformRootsWithoutPass[i].updateTransformListenersAndCompute( false, false, this._frameId, false );
      }
      this._dirtyTransformRootsWithoutPass.length = 0;
    },
    
    // NOTE: to be replaced with a full stitching/update version
    buildTemporaryDisplay: function() {
      // validate bounds for everywhere that could trigger bounds listeners. we want to flush out any changes, so that we can call validateBounds()
      // from code below without triggering side effects (we assume that we are not reentrant).
      this._rootNode.validateWatchedBounds();
      
      // throw new Error( 'TODO: replace with actual stitching' );
      this._baseInstance = createInstance( this, new scenery.Trail( this._rootNode ), scenery.RenderState.RegularState.createRootState( this._rootNode ), null );
      this.markTransformRootDirty( this._baseInstance, false ); // marks the transform root as dirty (since it is)
      
      this._rootBackbone = new scenery.BackboneBlock( this._baseInstance, scenery.bitmaskSupportsDOM, true, this._domElement );
      
      if ( assertSlow ) { this._baseInstance.audit( this._frameId ); }
      
      // pre-repaint phase: update relative transform information for listeners (notification) and precomputation where desired
      this.updateDirtyTransformRoots();
      
      if ( assertSlow ) { this._baseInstance.audit( this._frameId ); }
      
      // throw new Error( 'TODO: repaint phase (painting)' );
      
      // throw new Error( 'TODO: update cursor' );
      
      this._frameId++;
    }
    
    // TODO: add updateTemporaryDisplay?
  } );
  
  return Display;
} );
