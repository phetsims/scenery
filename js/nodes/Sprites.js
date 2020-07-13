// Copyright 2019-2020, University of Colorado Boulder

/**
 * Meant for displaying a large amount of high-performance instances of sprites.
 * See https://github.com/phetsims/scenery/issues/990 for more information.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Shape from '../../../kite/js/Shape.js';
import merge from '../../../phet-core/js/merge.js';
import SpritesCanvasDrawable from '../display/drawables/SpritesCanvasDrawable.js';
import SpritesWebGLDrawable from '../display/drawables/SpritesWebGLDrawable.js';
import Renderer from '../display/Renderer.js';
import scenery from '../scenery.js';
import Node from './Node.js';

class Sprites extends Node {
  /**
   * @param {Object} [options] Passed to Node
   */
  constructor( options ) {

    options = merge( {
      // {Array.<Sprite>} - Provide a fixed set of Sprite objects that will be used for this node. Currently it
      // cannot be modified after construction for simplicity of the implementation.
      sprites: [],

      // {Array.<SpriteInstance>} - A reference to an Array of instances. This array can be (and should be)
      // manipulated to adjust the displayed instances (their sprites/transforms/etc.). After this has been adjusted,
      // invalidatePaint() should be called on the Sprites node.
      spriteInstances: [],

      // {boolean} - Whether individual sprites will be hit-tested to determine what is a contained point (for hit
      // testing, etc.). If false, the canvasBounds will be used for hit testing.
      hitTestSprites: false,

      // Sets the node's default renderer to WebGL (as we'll generally want that when using this type)
      renderer: 'webgl'
    }, options );

    super();

    // @private {Array.<Sprite>}
    this._sprites = options.sprites;

    // @private {Array.<SpriteInstance>}
    this._spriteInstances = options.spriteInstances;

    // @private {boolean}
    this._hitTestSprites = options.hitTestSprites;

    // WebGL and Canvas are supported.
    this.setRendererBitmask( Renderer.bitmaskCanvas | Renderer.bitmaskWebGL );

    this.mutate( options );
  }

  /**
   * Sets the bounds that are used for layout/repainting.
   * @public
   *
   * These bounds should always cover at least the area where the Sprites will draw in. If this is violated, this
   * node may be partially or completely invisible in Scenery's output.
   *
   * @param {Bounds2} selfBounds
   */
  setCanvasBounds( selfBounds ) {
    this.invalidateSelf( selfBounds );
  }
  set canvasBounds( value ) { this.setCanvasBounds( value ); }

  /**
   * Returns the previously-set canvasBounds, or Bounds2.NOTHING if it has not been set yet.
   * @public
   *
   * @returns {Bounds2}
   */
  getCanvasBounds() {
    return this.getSelfBounds();
  }
  get canvasBounds() { return this.getCanvasBounds(); }

  /**
   * Draws the current Node's self representation, assuming the wrapper's Canvas context is already in the local
   * coordinate frame of this node.
   * @protected
   * @override
   *
   * @param {CanvasContextWrapper} wrapper
   * @param {Matrix3} matrix - The transformation matrix already applied to the context.
   */
  canvasPaintSelf( wrapper, matrix ) {
    SpritesCanvasDrawable.prototype.paintCanvas( wrapper, this, matrix );
  }

  /**
   * Creates a Canvas drawable for this Sprites.
   * @public (scenery-internal)
   * @override
   *
   * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param {Instance} instance - Instance object that will be associated with the drawable
   * @returns {CanvasSelfDrawable}
   */
  createCanvasDrawable( renderer, instance ) {
    return SpritesCanvasDrawable.createFromPool( renderer, instance );
  }

  /**
   * Creates a WebGL drawable for this Image.
   * @public (scenery-internal)
   * @override
   *
   * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param {Instance} instance - Instance object that will be associated with the drawable
   * @returns {WebGLSelfDrawable}
   */
  createWebGLDrawable( renderer, instance ) {
    return SpritesWebGLDrawable.createFromPool( renderer, instance );
  }

  /**
   * Computes whether the provided point is "inside" (contained) in this Node's self content, or "outside".
   * @protected
   * @override
   *
   * @param {Vector2} point - Considered to be in the local coordinate frame
   * @returns {boolean}
   */
  containsPointSelf( point ) {
    const inBounds = super.containsPointSelf( point );
    if ( !inBounds ) {
      return false;
    }

    if ( this._hitTestSprites ) {
      return !!this.getSpriteInstanceFromPoint( point );
    }
    else {
      return true;
    }
  }

  /**
   * Finds which sprite instance is on top under a certain point (or null if none are).
   * @public
   *
   * @param {Vector2} point
   * @returns {SpriteInstance|null}
   */
  getSpriteInstanceFromPoint( point ) {
    for ( let i = this._spriteInstances.length - 1; i >= 0; i-- ) {
      if ( this._spriteInstances[ i ].containsPoint( point ) ) {
        return this._spriteInstances[ i ];
      }
    }
    return null;
  }

  /**
   * Returns a Shape that represents the area covered by containsPointSelf.
   * @public
   * @override
   *
   * @returns {Shape}
   */
  getSelfShape() {
    if ( this._hitTestSprites ) {
      return Shape.union( this._spriteInstances.map( instance => instance.getShape() ) );
    }
    else {
      return Shape.bounds( this.selfBounds );
    }
  }

  /**
   * Whether this Node itself is painted (displays something itself).
   * @public
   * @override
   *
   * @returns {boolean}
   */
  isPainted() {
    // Always true for Sprites nodes
    return true;
  }

  /**
   * Should be called when this node needs to be repainted. When not called, Scenery assumes that this node does
   * NOT need to be repainted (although Scenery may repaint it due to other nodes needing to be repainted).
   * @public
   *
   * This sets a "dirty" flag, so that it will be repainted the next time it would be displayed.
   */
  invalidatePaint() {
    const stateLen = this._drawables.length;
    for ( let i = 0; i < stateLen; i++ ) {
      this._drawables[ i ].markDirty();
    }
  }
}

/**
 * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
 * order they will be evaluated in.
 * @protected
 *
 * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
 *       cases that may apply.
 */
Sprites.prototype._mutatorKeys = [ 'canvasBounds' ].concat( Node.prototype._mutatorKeys );

scenery.register( 'Sprites', Sprites );
export default Sprites;