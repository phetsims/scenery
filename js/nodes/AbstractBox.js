//  Copyright 2002-2014, University of Colorado Boulder

/**
 * AbstractBox is the parent for VBox and HBox, which arranges child nodes.
 * See https://github.com/phetsims/scenery/issues/116
 *
 * @author Sam Reid
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Node = require( 'SCENERY/nodes/Node' );

  /**
   * Main constructor for AbstractBox.
   *
   * @param {object} options Same as Node.constructor.options with the following additions:
   *
   * spacing: can be a number or a function.  If a number, then it will be the spacing between each object.
   *              If a function, then the function will have the signature function(a,b){} which returns the spacing between adjacent pairs of items.
   * align:   How to line up the items, defaults to 'center'.
   *
   * @param {string} type 'vertical' or 'horizontal'
   * @param {function} layoutFunction the function to layout out the nodes (different for HBox and VBox)
   *
   * @constructor
   */
  scenery.AbstractBox = function AbstractBox( type, layoutFunction, options ) {
    assert && assert( type === 'vertical' || type === 'horizontal' );

    this.type = type;
    this.layoutFunction = layoutFunction;
    this.boundsListener = this.updateLayout.bind( this );

    // ensure we have a parameter object
    this.options = _.extend( {
      // defaults
      spacing: function() { return 0; },
      align: 'center',

      //By default, update the layout when children are added/removed/resized, see #116
      resize: true
    }, options );

    if ( typeof this.options.spacing === 'number' ) {
      var spacingConstant = this.options.spacing;
      this.options.spacing = function() { return spacingConstant; };
    }

    Node.call( this );

    //See HBox.js
    this.inited = false;
    this.mutate( this.options );
    this.inited = true;
  };
  var AbstractBox = scenery.AbstractBox;

  inherit( Node, AbstractBox, {
    updateLayout: function() {
      if ( !this.updatingLayout ) {
        //Bounds of children are changed in updateLayout, we don't want to stackoverflow so bail if already updating layout
        this.updatingLayout = true;
        this.layoutFunction.call( this );
        this.updatingLayout = false;
      }
    }
  } );

  //Override the child mutators to updateLayout
  //Have to listen to the child bounds individually because there are a number of possible ways to change the child
  //bounds without changing the overall bounds.
  var overrides = ['insertChild', 'removeChildWithIndex'];
  overrides.forEach( function( override ) {

    //Support up to two args for overrides
    AbstractBox.prototype[override] = function( arg1, arg2 ) {

      //Remove event listeners from any nodes (will be added back later if the node was not removed)
      var abstractBox = this;
      if ( this.options.resize ) {
        this.children.forEach( function( child ) {
          if ( child.containsEventListener( 'bounds', abstractBox.boundsListener ) ) {
            child.removeEventListener( 'bounds', abstractBox.boundsListener );
          }
        } );
      }

      //Super call
      Node.prototype[override].call( this, arg1, arg2 );

      //Update the layout if it should be dynamic
      if ( this.options.resize || !this.inited ) {
        this.updateLayout();
      }

      //Add event listeners for any current children (if it should be dynamic)
      if ( this.options.resize ) {
        this.children.forEach( function( child ) {
          if ( !child.containsEventListener( 'bounds', abstractBox.boundsListener ) ) {
            child.addEventListener( 'bounds', abstractBox.boundsListener );
          }
        } );
      }
    };
  } );

  return AbstractBox;
} );