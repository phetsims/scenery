import Vector2 from "../../../dot/js/Vector2";
import {PhetioObjectOptions} from "../../../tandem/js/PhetioObject.js";

export type NodeOptions = {
  cursor: string;
  maxWidth: number;
  centerX: number;
  centerY: number;
  x: number;
  y: number;
  rotation: number;
} & PhetioObjectOptions;

export default class Node {
  constructor( options?: Partial<NodeOptions> );

  addChild( node: Node );

  scale( s: number );
  scale( sx: number, sy: number );

  set translation( t: Vector2 );

  set rotation( r: number );

  addInputListener( listener: any ): void;

  get height(): number;

  get width(): number;
}