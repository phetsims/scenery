// Copyright 2014-2023, University of Colorado Boulder

/**
 * An abstract node (should be subtyped) that is drawn by user-provided custom WebGL code.
 *
 * The region that can be drawn in is handled manually, by controlling the canvasBounds property of this WebGLNode.
 * Any regions outside of the canvasBounds will not be guaranteed to be drawn. This can be set with canvasBounds in the
 * constructor, or later with node.canvasBounds = bounds or setCanvasBounds( bounds ).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Sam Reid (PhET Interactive Simulations)
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import Matrix3 from '../../../dot/js/Matrix3.js';
import Vector2 from '../../../dot/js/Vector2.js';
import { Shape } from '../../../kite/js/imports.js';
import { CanvasContextWrapper, Instance, Node, NodeOptions, Renderer, scenery, Utils, WebGLNodeDrawable, WebGLSelfDrawable } from '../imports.js';

const WEBGL_NODE_OPTION_KEYS = [
  'canvasBounds' // {Bounds2} - Sets the available Canvas bounds that content will show up in. See setCanvasBounds()
];

type SelfOptions = {
  canvasBounds?: Bounds2;
};

export type WebGLNodeOptions = SelfOptions & NodeOptions;
export type WebGLNodePainterResult = 0 | 1;

export type WebGLNodePainter = {
  paint: ( modelViewMatrix: Matrix3, projectionMatrix: Matrix3 ) => WebGLNodePainterResult;
  dispose: () => void;
};
// NOTE: the `node` will be the `this` type, but there doesn't seem to be a good way to annotate that
type WebGLNodePainterType = new ( gl: WebGLRenderingContext, node: never ) => WebGLNodePainter;

export default abstract class WebGLNode extends Node {

  // Used to create the painters
  private painterType: WebGLNodePainterType;

  /**
   *
   * It is required to pass a canvasBounds option and/or keep canvasBounds such that it will cover the entirety of the
   * Node. This will also set its self bounds.
   *
   * A "Painter" type should be passed to the constructor. It will be responsible for creating individual "painters"
   * that are used with different WebGL contexts to paint. This is helpful, since each context will need to have its
   * own buffers/textures/etc.
   *
   * painterType will be called with new painterType( gl, node ). Should contain the following methods:
   *
   * paint( modelViewMatrix, projectionMatrix )
   *   {Matrix3} modelViewMatrix - Transforms from the node's local coordinate frame to Scenery's global coordinate
   *                               frame.
   *   {Matrix3} projectionMatrix - Transforms from the global coordinate frame to normalized device coordinates.
   *   Returns either WebGLNode.PAINTED_NOTHING or WebGLNode.PAINTED_SOMETHING.
   * dispose()
   *
   * NOTE: If any alpha values are non-1, please note that Scenery's canvases uses blending/settings for premultiplied
   * alpha. This means that if you want a color to look like (r,g,b,a), the value passed to gl_FragColor should be
   * (r/a,g/a,b/a,a).
   *
   * @param painterType - The type (constructor) for the painters that will be used for this node.
   * @param [options] - WebGLNode-specific options are documented in LINE_OPTION_KEYS above, and can be
   *                    provided along-side options for Node
   */
  public constructor( painterType: WebGLNodePainterType, options?: WebGLNodeOptions ) {
    super( options );

    // Only support rendering in WebGL
    this.setRendererBitmask( Renderer.bitmaskWebGL );

    this.painterType = painterType;
  }

  /**
   * Sets the bounds that are used for layout/repainting.
   *
   * These bounds should always cover at least the area where the WebGLNode will draw in. If this is violated, this
   * node may be partially or completely invisible in Scenery's output.
   */
  public setCanvasBounds( selfBounds: Bounds2 ): this {
    this.invalidateSelf( selfBounds );

    return this;
  }

  public set canvasBounds( value: Bounds2 ) { this.setCanvasBounds( value ); }

  public get canvasBounds(): Bounds2 { return this.getCanvasBounds(); }

  /**
   * Returns the previously-set canvasBounds, or Bounds2.NOTHING if it has not been set yet.
   */
  public getCanvasBounds(): Bounds2 {
    return this.getSelfBounds();
  }

  /**
   * Whether this Node itself is painted (displays something itself).
   */
  public override isPainted(): boolean {
    // Always true for WebGL nodes
    return true;
  }

  /**
   * Should be called when this node needs to be repainted. When not called, Scenery assumes that this node does
   * NOT need to be repainted (although Scenery may repaint it due to other nodes needing to be repainted).
   *
   * This sets a "dirty" flag, so that it will be repainted the next time it would be displayed.
   */
  public invalidatePaint(): void {
    const stateLen = this._drawables.length;
    for ( let i = 0; i < stateLen; i++ ) {
      this._drawables[ i ].markDirty();
    }
  }

  /**
   * Computes whether the provided point is "inside" (contained) in this Node's self content, or "outside".
   *
   * If WebGLNode subtypes want to support being picked or hit-tested, it should override this function.
   *
   * @param point - Considered to be in the local coordinate frame
   */
  public override containsPointSelf( point: Vector2 ): boolean {
    return false;
  }

  /**
   * Returns a Shape that represents the area covered by containsPointSelf.
   */
  public override getSelfShape(): Shape {
    return new Shape();
  }

  /**
   * Draws the current Node's self representation, assuming the wrapper's Canvas context is already in the local
   * coordinate frame of this node. (scenery-internal)
   *
   * @param wrapper
   * @param matrix - The transformation matrix already applied to the context.
   */
  protected override canvasPaintSelf( wrapper: CanvasContextWrapper, matrix: Matrix3 ): void {
    // TODO: see https://github.com/phetsims/scenery/issues/308
    assert && assert( false, 'unimplemented: canvasPaintSelf in WebGLNode' );
  }

  /**
   * Renders this Node only (its self) into the Canvas wrapper, in its local coordinate frame.
   *
   * @param wrapper
   * @param matrix - The current transformation matrix associated with the wrapper
   */
  public override renderToCanvasSelf( wrapper: CanvasContextWrapper, matrix: Matrix3 ): void {
    const width = wrapper.canvas.width;
    const height = wrapper.canvas.height;

    // TODO: Can we reuse the same Canvas? That might save some context creations?
    const scratchCanvas = document.createElement( 'canvas' );
    scratchCanvas.width = width;
    scratchCanvas.height = height;
    const contextOptions = {
      antialias: true,
      preserveDrawingBuffer: true // so we can get the data and render it to the Canvas
    };
    const gl = ( scratchCanvas.getContext( 'webgl', contextOptions ) || scratchCanvas.getContext( 'experimental-webgl', contextOptions ) ) as WebGLRenderingContext;
    Utils.applyWebGLContextDefaults( gl ); // blending, etc.

    const projectionMatrix = new Matrix3().rowMajor(
      2 / width, 0, -1,
      0, -2 / height, 1,
      0, 0, 1 );
    gl.viewport( 0, 0, width, height );

    const PainterType = this.painterType;
    const painter = new PainterType( gl, this as never );

    painter.paint( matrix, projectionMatrix );
    painter.dispose();

    projectionMatrix.freeToPool();

    gl.flush();

    wrapper.context.setTransform( 1, 0, 0, 1, 0, 0 ); // identity
    wrapper.context.drawImage( scratchCanvas, 0, 0 );
    wrapper.context.restore();
  }

  /**
   * Creates a WebGL drawable for this WebGLNode. (scenery-internal)
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   */
  public override createWebGLDrawable( renderer: number, instance: Instance ): WebGLSelfDrawable {
    // @ts-expect-error TODO: pooling
    return WebGLNodeDrawable.createFromPool( renderer, instance );
  }

  public override mutate( options?: WebGLNodeOptions ): this {
    return super.mutate( options );
  }

  // Return code from painter.paint() when nothing was painted to the WebGL context.
  public static readonly PAINTED_NOTHING = 0 as const;

  // Return code from painter.paint() when something was painted to the WebGL context.
  public static readonly PAINTED_SOMETHING = 1 as const;
}

/**
 * {Array.<string>} - String keys for all the allowed options that will be set by node.mutate( options ), in the
 * order they will be evaluated in.
 *
 * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
 *       cases that may apply.
 */
WebGLNode.prototype._mutatorKeys = WEBGL_NODE_OPTION_KEYS.concat( Node.prototype._mutatorKeys );

scenery.register( 'WebGLNode', WebGLNode );
