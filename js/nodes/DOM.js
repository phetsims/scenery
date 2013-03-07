// Copyright 2002-2012, University of Colorado

/**
 * DOM nodes. Currently lightweight handling
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var Bounds2 = require( 'DOT/Bounds2' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Node = require( 'SCENERY/nodes/Node' ); // DOM inherits from Node
  var Renderer = require( 'SCENERY/layers/Renderer' );
  var objectCreate = require( 'SCENERY/util/Util' ).objectCreate;
  
  scenery.DOM = function( element, options ) {
    options = options || {};
    
    // unwrap from jQuery if that is passed in, for consistency
    if ( element && element.jquery ) {
      element = element[0];
    }
    
    this._element = element;
    this._$element = $( element );
    this._$element.css( 'position', 'absolute' );
    this._$element.css( 'left', 0 );
    this._$element.css( 'top', 0 );
    
    this.attachedToDOM = false;
    
    // so that the mutator will call setElement()
    options.element = element;
    
    // create a temporary container attached to the DOM tree (not a fragment) so that we can properly set initial bounds
    var temporaryContainer = this.wrapInTemporaryContainer();
    
    // will set the element after initializing
    Node.call( this, options );
    
    // now don't memory-leak our container
    this.destroyTemporaryContainer( temporaryContainer );
  };
  var DOM = scenery.DOM;
  
  DOM.prototype = objectCreate( Node.prototype );
  DOM.prototype.constructor = DOM;
  
  DOM.prototype.invalidatePaint = function( bounds ) {
    Node.prototype.invalidatePaint.call( this, bounds );
  };
  
  // needs to be attached to the DOM tree for this to work
  DOM.prototype.calculateDOMBounds = function() {
    return new Bounds2( 0, 0, this._$element.width(), this._$element.height() );
  };
  
  DOM.prototype.invalidateDOM = function() {
    // TODO: do we need to reset the CSS transform to get the proper bounds?
    
    if ( !this.attachedToDOM ) {
      // create a temporary container attached to the DOM tree (not a fragment) so that we can properly get the bounds
      var temporaryContainer = this.wrapInTemporaryContainer();
      
      this.invalidateSelf( this.calculateDOMBounds() );
      
      // now don't memory-leak our container
      this.destroyTemporaryContainer( temporaryContainer );
    } else {
      this.invalidateSelf( this.calculateDOMBounds() );
    }
  };
  
  DOM.prototype.addToDOMLayer = function( domLayer ) {
    this.attachedToDOM = true;
    
    // TODO: find better way to handle non-jquery and jquery-wrapped getters for the container. direct access for now ()
    domLayer.$div.append( this._element );
    
    // recompute the bounds
    this.invalidateDOM();
  };
  
  DOM.prototype.removeFromDOMLayer = function( domLayer ) {
    domLayer.$div.remove( this._element );
    this.attachedToDOM = false;
  };
  
  DOM.prototype.updateCSSTransform = function( transform ) {
    this._$element.css( transform.getMatrix().cssTransformStyles() );
  };
  
  DOM.prototype.wrapInTemporaryContainer = function() {
    // create a temporary container attached to the DOM tree (not a fragment) so that we can properly set initial bounds
    var temporaryContainer = document.createElement( 'div' );
    $( temporaryContainer ).css( {
      display: 'hidden',
      padding: '0 !important',
      margin: '0 !important',
      position: 'absolute',
      left: 0,
      top: 0,
      width: 65535,
      height: 65535
    } );
    temporaryContainer.appendChild( this._element );
    document.body.appendChild( temporaryContainer );
    return temporaryContainer;
  };
  
  DOM.prototype.destroyTemporaryContainer = function( temporaryContainer ) {
    document.body.removeChild( temporaryContainer );
    if ( this._element.parentNode === temporaryContainer ) {
      temporaryContainer.removeChild( this._element );
    }
  };
  
  DOM.prototype.hasSelf = function() {
    return true;
  };
  
  DOM.prototype.setElement = function( element ) {
    this._element = element;
    
    // TODO: bounds issue, since this will probably set to empty bounds and thus a repaint may not draw over it
    this.invalidateDOM();
    
    return this; // allow chaining
  };
  
  DOM.prototype.getElement = function() {
    return this._element;
  };
  
  DOM.prototype._mutatorKeys = [ 'element' ].concat( Node.prototype._mutatorKeys );
  
  DOM.prototype._supportedRenderers = [ Renderer.DOM ];
  
  Object.defineProperty( DOM.prototype, 'element', { set: DOM.prototype.setElement, get: DOM.prototype.getElement } );
  
  return DOM;
} );


